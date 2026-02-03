"""
Advanced JIT Optimizer for MathViz Compiler.

This module provides intelligent decision-making for Numba JIT compilation,
analyzing function characteristics to determine optimal compilation strategies.

The optimizer examines:
- Function purity and side effects
- Loop patterns and parallelization potential
- Memory access patterns and cache behavior
- Arithmetic intensity and vectorization opportunities
- Numba compatibility constraints

Key Features:
1. Smart JIT mode selection (none, jit, njit, vectorize, etc.)
2. Loop optimization hints (prange, fusion, tiling)
3. Vectorization detection for SIMD opportunities
4. Cache-aware optimization suggestions
5. Cost model for JIT compilation decisions

References:
- Numba JIT documentation: https://numba.readthedocs.io/
- Intel Intrinsics Guide for vectorization patterns
- "Optimizing Compilers for Modern Architectures" by Allen & Kennedy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mathviz.compiler.complexity_analyzer import ComplexityInfo
    from mathviz.compiler.parallel_analyzer import LoopAnalysis
    from mathviz.compiler.purity_analyzer import PurityInfo

from mathviz.compiler.ast_nodes import (
    AssignmentStatement,
    ASTNode,
    BaseASTVisitor,
    BinaryExpression,
    BinaryOperator,
    CallExpression,
    CompoundAssignment,
    ConditionalExpression,
    Expression,
    FloatLiteral,
    ForStatement,
    FunctionDef,
    GenericType,
    Identifier,
    IfStatement,
    IndexExpression,
    IntegerLiteral,
    JitMode,
    JitOptions,
    LetStatement,
    RangeExpression,
    ReturnStatement,
    SimpleType,
    # Types
    TypeAnnotation,
    UnaryExpression,
    WhileStatement,
)
from mathviz.utils.errors import SourceLocation

# =============================================================================
# JIT Strategy Types
# =============================================================================


class JitStrategy(Enum):
    """
    JIT compilation strategies supported by the optimizer.

    Each strategy corresponds to a different Numba decorator or
    compilation mode with varying characteristics:

    - NONE: Skip JIT compilation entirely
    - JIT: Standard @jit with object mode fallback
    - NJIT: No-python @njit (fastest, most restrictive)
    - NJIT_PARALLEL: @njit with parallel=True
    - NJIT_FASTMATH: @njit with fastmath=True
    - VECTORIZE: @vectorize for element-wise ufuncs
    - GUVECTORIZE: @guvectorize for generalized ufuncs
    - STENCIL: @stencil for neighborhood operations
    - CUDA: GPU compilation (future feature)
    """

    NONE = "none"
    JIT = "jit"
    NJIT = "njit"
    NJIT_PARALLEL = "njit_parallel"
    NJIT_FASTMATH = "njit_fastmath"
    VECTORIZE = "vectorize"
    GUVECTORIZE = "guvectorize"
    STENCIL = "stencil"
    CUDA = "cuda"

    @property
    def requires_nopython(self) -> bool:
        """Check if strategy requires nopython mode."""
        return self in {
            JitStrategy.NJIT,
            JitStrategy.NJIT_PARALLEL,
            JitStrategy.NJIT_FASTMATH,
            JitStrategy.VECTORIZE,
            JitStrategy.GUVECTORIZE,
            JitStrategy.STENCIL,
        }

    @property
    def supports_parallel(self) -> bool:
        """Check if strategy supports parallel execution."""
        return self in {
            JitStrategy.NJIT_PARALLEL,
            JitStrategy.VECTORIZE,
            JitStrategy.GUVECTORIZE,
        }


class MemoryAccessPattern(Enum):
    """Classification of memory access patterns."""

    SEQUENTIAL = auto()  # arr[i], arr[i+1], ...
    STRIDED = auto()  # arr[i*k], arr[i*k + c], ...
    RANDOM = auto()  # arr[indices[i]], unpredictable
    BROADCAST = auto()  # Same element accessed by all iterations
    GATHER = auto()  # Irregular read pattern
    SCATTER = auto()  # Irregular write pattern


class LoopPattern(Enum):
    """Classification of loop computation patterns."""

    MAP = auto()  # arr[i] = f(arr[i])
    REDUCTION = auto()  # result = f(result, arr[i])
    SCAN = auto()  # arr[i] = f(arr[i-1])
    STENCIL = auto()  # arr[i] = f(arr[i-1], arr[i], arr[i+1])
    HISTOGRAM = auto()  # bins[arr[i]] += 1
    TRANSPOSE = auto()  # out[j,i] = in[i,j]
    MATMUL = auto()  # c[i,j] = sum(a[i,k] * b[k,j])
    DOT_PRODUCT = auto()  # result = sum(a[i] * b[i])
    UNKNOWN = auto()  # Cannot classify


# =============================================================================
# Analysis Result Data Structures
# =============================================================================


@dataclass(slots=True)
class JitDecision:
    """
    The optimizer's decision about JIT compilation for a function.

    Attributes:
        strategy: The recommended JIT strategy
        options: Numba decorator options (cache, parallel, fastmath, etc.)
        confidence: Confidence score from 0.0 to 1.0
        reasons: List of reasons supporting this decision
        warnings: Potential issues or caveats
        estimated_speedup: Estimated speedup factor (if measurable)
        alternative_strategies: Viable alternative approaches
    """

    strategy: JitStrategy = JitStrategy.NONE
    options: dict = field(default_factory=dict)
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimated_speedup: float | None = None
    alternative_strategies: list[JitStrategy] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"JIT Decision: {self.strategy.value}"]
        lines.append(f"  Confidence: {self.confidence:.1%}")
        if self.estimated_speedup:
            lines.append(f"  Estimated Speedup: {self.estimated_speedup:.1f}x")
        if self.options:
            lines.append(f"  Options: {self.options}")
        if self.reasons:
            lines.append("  Reasons:")
            for reason in self.reasons:
                lines.append(f"    - {reason}")
        if self.warnings:
            lines.append("  Warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        return "\n".join(lines)

    def to_jit_options(self) -> JitOptions:
        """Convert decision to JitOptions for code generation."""
        mode_map = {
            JitStrategy.NONE: JitMode.NONE,
            JitStrategy.JIT: JitMode.JIT,
            JitStrategy.NJIT: JitMode.NJIT,
            JitStrategy.NJIT_PARALLEL: JitMode.NJIT,
            JitStrategy.NJIT_FASTMATH: JitMode.NJIT,
            JitStrategy.VECTORIZE: JitMode.VECTORIZE,
            JitStrategy.GUVECTORIZE: JitMode.GUVECTORIZE,
            JitStrategy.STENCIL: JitMode.STENCIL,
            JitStrategy.CUDA: JitMode.NJIT,  # Fallback
        }

        return JitOptions(
            mode=mode_map.get(self.strategy, JitMode.NONE),
            nopython=self.options.get("nopython", True),
            nogil=self.options.get("nogil", False),
            cache=self.options.get("cache", True),
            parallel=self.options.get("parallel", False),
            fastmath=self.options.get("fastmath", False),
            boundscheck=self.options.get("boundscheck", False),
            inline=self.options.get("inline", "never"),
        )


@dataclass(slots=True)
class VectorizableOp:
    """Represents an operation that can be vectorized."""

    expression: Expression
    element_type: str
    operation: str
    estimated_simd_lanes: int = 4
    location: SourceLocation | None = None


@dataclass(slots=True)
class LoopTransformation:
    """A suggested loop transformation for better performance."""

    loop: ForStatement
    transformation: str  # "prange", "fuse", "tile", "interchange", "unroll"
    description: str
    expected_benefit: str
    prerequisites: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReductionInfo:
    """Information about a detected reduction pattern."""

    variable: str
    operator: str  # "sum", "product", "max", "min", "and", "or"
    init_value: Expression | None = None
    is_parallel_safe: bool = True


@dataclass(slots=True)
class TilingInfo:
    """Information about loop tiling potential."""

    loop_variable: str
    suggested_tile_size: int
    estimated_cache_benefit: float
    data_reuse_factor: float


@dataclass(slots=True)
class MemoryPattern:
    """Analysis of memory access patterns in a function."""

    pattern: MemoryAccessPattern
    arrays_accessed: set[str] = field(default_factory=set)
    stride: int = 1  # Stride in elements
    is_cache_friendly: bool = True
    working_set_estimate: int = 0  # In bytes
    suggestions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VectorizationInfo:
    """Vectorization analysis results."""

    is_vectorizable: bool = False
    vectorizable_ops: list[VectorizableOp] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    recommended_strategy: str | None = None  # "numpy_vectorize", "numba_vectorize", "ufunc"
    estimated_simd_speedup: float = 1.0


@dataclass(slots=True)
class CacheHints:
    """Cache optimization hints for a function."""

    working_set_size: int = 0  # Estimated bytes
    fits_l1: bool = True
    fits_l2: bool = True
    fits_l3: bool = True
    cache_unfriendly_patterns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# =============================================================================
# Numba Compatibility Constants
# =============================================================================


# Types compatible with Numba nopython mode
NUMBA_COMPATIBLE_TYPES: frozenset[str] = frozenset(
    {
        # Scalar types
        "Int",
        "Float",
        "Bool",
        "int",
        "float",
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
        # NumPy array types
        "Array",
        "Vec",
        "Mat",
        "Matrix",
        "NDArray",
        "ndarray",
        "np.ndarray",
    }
)

# Types that prevent nopython mode
NUMBA_INCOMPATIBLE_TYPES: frozenset[str] = frozenset(
    {
        "String",
        "str",
        "Set",
        "set",
        "Dict",
        "dict",
        "Optional",
        "Result",
        "Any",
        "object",
    }
)

# Built-in functions supported in nopython mode
NUMBA_SUPPORTED_BUILTINS: frozenset[str] = frozenset(
    {
        # Math operations
        "abs",
        "min",
        "max",
        "sum",
        "pow",
        "round",
        "int",
        "float",
        "bool",
        "complex",
        # Container operations
        "len",
        "range",
        "enumerate",
        "zip",
        # Math functions (will be replaced with np.*)
        "sqrt",
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "log10",
        "log2",
        "floor",
        "ceil",
        "fabs",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "degrees",
        "radians",
        # NumPy array creation
        "zeros",
        "ones",
        "empty",
        "arange",
        "linspace",
        "eye",
        "full",
        "zeros_like",
        "ones_like",
        "empty_like",
        # NumPy array operations
        "dot",
        "matmul",
        "transpose",
        "reshape",
        "flatten",
        "ravel",
        "concatenate",
        "stack",
        "vstack",
        "hstack",
        "mean",
        "std",
        "var",
        "prod",
        "argmax",
        "argmin",
        "argsort",
        "sort",
        "clip",
        "where",
        "nonzero",
        "unique",
        "diff",
        "cumsum",
        "cumprod",
    }
)

# Functions that prevent JIT compilation
JIT_BLOCKING_FUNCTIONS: frozenset[str] = frozenset(
    {
        "print",
        "println",
        "input",
        "open",
        "read_file",
        "write_file",
        "append_file",
        "format",
        "repr",
        "str",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
    }
)

# Operations that benefit from fastmath
FASTMATH_BENEFICIAL_OPS: frozenset[str] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "pow",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
    }
)

# Typical cache sizes for reference (in bytes)
L1_CACHE_SIZE = 32 * 1024  # 32 KB
L2_CACHE_SIZE = 256 * 1024  # 256 KB
L3_CACHE_SIZE = 8 * 1024 * 1024  # 8 MB

# Type size estimates (in bytes)
TYPE_SIZES: dict[str, int] = {
    "Int": 8,
    "Float": 8,
    "Bool": 1,
    "int": 8,
    "float": 8,
    "bool": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


# =============================================================================
# JIT Analyzer - Main Analysis Class
# =============================================================================


class JitAnalyzer:
    """
    Analyzes functions to determine optimal JIT compilation strategy.

    This analyzer examines function characteristics including:
    - Parameter and return types
    - Loop patterns and iteration counts
    - Memory access patterns
    - Arithmetic intensity
    - Function calls and dependencies

    Example:
        analyzer = JitAnalyzer()
        decision = analyzer.analyze_function(func_def)
        if decision.strategy != JitStrategy.NONE:
            print(f"Recommend {decision.strategy.value} with {decision.confidence:.0%} confidence")
    """

    def __init__(
        self,
        purity_info: dict[str, PurityInfo] | None = None,
        complexity_info: dict[str, ComplexityInfo] | None = None,
        parallel_info: dict[str, list[LoopAnalysis]] | None = None,
    ) -> None:
        """
        Initialize the JIT analyzer.

        Args:
            purity_info: Pre-computed purity analysis results
            complexity_info: Pre-computed complexity analysis results
            parallel_info: Pre-computed parallelization analysis results
        """
        self._purity_info = purity_info or {}
        self._complexity_info = complexity_info or {}
        self._parallel_info = parallel_info or {}

        # Internal analyzers
        self._loop_optimizer = LoopOptimizer()
        self._vectorization_analyzer = VectorizationAnalyzer()
        self._cache_optimizer = CacheOptimizer()
        self._cost_model = CostModel()

    def analyze_function(self, func: FunctionDef) -> JitDecision:
        """
        Analyze a function and decide the optimal JIT strategy.

        This is the main entry point for JIT analysis. It examines
        all aspects of the function and produces a recommendation.

        Args:
            func: The function definition AST node

        Returns:
            JitDecision containing the recommended strategy and rationale
        """
        decision = JitDecision()

        # Step 1: Check if already has explicit JIT decorator
        if func.jit_options.mode != JitMode.NONE:
            decision.strategy = self._jit_mode_to_strategy(func.jit_options.mode)
            decision.confidence = 1.0
            decision.reasons.append("Function has explicit JIT decorator")
            decision.options = self._extract_existing_options(func.jit_options)
            return decision

        # Step 2: Check basic Numba compatibility
        is_compatible, compatibility_reasons = self._check_numba_compatibility(func)
        if not is_compatible:
            decision.strategy = JitStrategy.NONE
            decision.confidence = 0.9
            decision.reasons = ["Not Numba-compatible"] + compatibility_reasons
            return decision

        # Step 3: Check purity (side effects prevent JIT)
        if func.name in self._purity_info:
            purity = self._purity_info[func.name]
            if purity.has_io():
                decision.strategy = JitStrategy.NONE
                decision.confidence = 0.95
                decision.reasons.append("Function has I/O operations")
                decision.warnings.append("Extract pure computation into separate function for JIT")
                return decision
            if purity.has_manim_calls():
                decision.strategy = JitStrategy.NONE
                decision.confidence = 0.95
                decision.reasons.append("Function contains Manim animation calls")
                return decision

        # Step 4: Analyze loop patterns
        loop_analysis = self._analyze_loop_patterns(func)

        # Step 5: Analyze memory access patterns
        memory_pattern = self._analyze_memory_access(func)

        # Step 6: Calculate arithmetic intensity
        math_intensity = self._analyze_math_intensity(func)

        # Step 7: Detect vectorization opportunities
        vectorization = self._vectorization_analyzer.analyze(func)

        # Step 8: Determine strategy based on analysis
        strategy, confidence, reasons = self._determine_strategy(
            func=func,
            loop_analysis=loop_analysis,
            memory_pattern=memory_pattern,
            math_intensity=math_intensity,
            vectorization=vectorization,
        )

        decision.strategy = strategy
        decision.confidence = confidence
        decision.reasons = reasons

        # Step 9: Determine optimal options
        decision.options = self._determine_options(
            func=func,
            strategy=strategy,
            loop_analysis=loop_analysis,
            math_intensity=math_intensity,
        )

        # Step 10: Estimate speedup
        decision.estimated_speedup = self._cost_model.estimate_jit_benefit(func)

        # Step 11: Add warnings if applicable
        self._add_warnings(decision, func, memory_pattern)

        # Step 12: Suggest alternatives
        decision.alternative_strategies = self._suggest_alternatives(
            strategy, loop_analysis, vectorization
        )

        return decision

    def _jit_mode_to_strategy(self, mode: JitMode) -> JitStrategy:
        """Convert JitMode to JitStrategy."""
        mapping = {
            JitMode.NONE: JitStrategy.NONE,
            JitMode.JIT: JitStrategy.JIT,
            JitMode.NJIT: JitStrategy.NJIT,
            JitMode.VECTORIZE: JitStrategy.VECTORIZE,
            JitMode.GUVECTORIZE: JitStrategy.GUVECTORIZE,
            JitMode.STENCIL: JitStrategy.STENCIL,
            JitMode.CFUNC: JitStrategy.NJIT,  # Similar to njit
        }
        return mapping.get(mode, JitStrategy.NONE)

    def _extract_existing_options(self, jit_opts: JitOptions) -> dict:
        """Extract options from existing JitOptions."""
        return {
            "nopython": jit_opts.nopython,
            "nogil": jit_opts.nogil,
            "cache": jit_opts.cache,
            "parallel": jit_opts.parallel,
            "fastmath": jit_opts.fastmath,
            "boundscheck": jit_opts.boundscheck,
        }

    def _check_numba_compatibility(self, func: FunctionDef) -> tuple[bool, list[str]]:
        """
        Check if a function is compatible with Numba JIT compilation.

        Returns:
            Tuple of (is_compatible, list_of_blocking_reasons)
        """
        reasons: list[str] = []

        # Check parameter types
        for param in func.parameters:
            if param.type_annotation:
                if not self._is_numba_compatible_type(param.type_annotation):
                    reasons.append(f"Parameter '{param.name}' has incompatible type")

        # Check return type
        if func.return_type and not self._is_numba_compatible_type(func.return_type):
            reasons.append("Return type is not Numba-compatible")

        # Check function body for incompatible constructs
        if func.body:
            body_checker = _NumbaCompatibilityVisitor()
            body_checker.visit(func.body)
            reasons.extend(body_checker.blocking_reasons)

        return len(reasons) == 0, reasons

    def _is_numba_compatible_type(self, type_ann: TypeAnnotation) -> bool:
        """Check if a type annotation is Numba-compatible."""
        if isinstance(type_ann, SimpleType):
            if type_ann.name in NUMBA_COMPATIBLE_TYPES:
                return True
            if type_ann.name in NUMBA_INCOMPATIBLE_TYPES:
                return False
            # Unknown types are assumed compatible (might be user arrays)
            return True
        elif isinstance(type_ann, GenericType):
            # List[Int], Array[Float], etc.
            if type_ann.base in {"List", "Array", "Vec", "Mat", "Matrix", "NDArray"}:
                return all(self._is_numba_compatible_type(arg) for arg in type_ann.type_args)
            if type_ann.base in {"Set", "Dict", "Optional"}:
                return False
        return True  # Default to compatible

    def _analyze_loop_patterns(self, func: FunctionDef) -> list[LoopAnalysis]:
        """Analyze loop patterns in the function."""
        # Use pre-computed parallel analysis if available
        if func.name in self._parallel_info:
            return self._parallel_info[func.name]

        # Otherwise, perform analysis
        return self._loop_optimizer.analyze_function_loops(func)

    def _analyze_memory_access(self, func: FunctionDef) -> MemoryPattern:
        """Analyze memory access patterns in the function."""
        analyzer = _MemoryAccessAnalyzer()
        if func.body:
            analyzer.visit(func.body)
        return analyzer.get_pattern()

    def _analyze_math_intensity(self, func: FunctionDef) -> float:
        """
        Calculate arithmetic intensity (operations per memory access).

        Higher values indicate compute-bound functions that benefit more from JIT.
        """
        analyzer = _ArithmeticIntensityAnalyzer()
        if func.body:
            analyzer.visit(func.body)
        return analyzer.get_intensity()

    def _determine_strategy(
        self,
        func: FunctionDef,
        loop_analysis: list[LoopAnalysis],
        memory_pattern: MemoryPattern,
        math_intensity: float,
        vectorization: VectorizationInfo,
    ) -> tuple[JitStrategy, float, list[str]]:
        """Determine the optimal JIT strategy based on all analysis."""
        reasons: list[str] = []

        # Check for vectorization opportunity
        if vectorization.is_vectorizable and len(func.parameters) == 1:
            # Unary function - good candidate for @vectorize
            return (
                JitStrategy.VECTORIZE,
                0.85,
                ["Function is element-wise with single input", "Ideal for @vectorize"],
            )

        # Check for parallel loops
        has_parallel_loops = any(la.can_use_prange for la in loop_analysis)
        has_reductions = any(la.reduction_vars for la in loop_analysis)

        if has_parallel_loops:
            reasons.append("Function has parallelizable loops")
            if has_reductions:
                reasons.append("Contains parallel-safe reduction patterns")
            return (JitStrategy.NJIT_PARALLEL, 0.90, reasons)

        # Check arithmetic intensity
        if math_intensity > 5.0:
            reasons.append(f"High arithmetic intensity ({math_intensity:.1f} ops/access)")
            if self._has_fastmath_opportunities(func):
                reasons.append("Contains transcendental functions benefiting from fastmath")
                return (JitStrategy.NJIT_FASTMATH, 0.85, reasons)
            return (JitStrategy.NJIT, 0.85, reasons)

        # Check for loops that benefit from JIT
        if loop_analysis:
            reasons.append("Function contains loops")
            if math_intensity > 1.0:
                reasons.append(f"Moderate arithmetic intensity ({math_intensity:.1f})")
                return (JitStrategy.NJIT, 0.80, reasons)
            else:
                reasons.append("Memory-bound computation")
                return (JitStrategy.JIT, 0.70, reasons)

        # Small function with no loops - still might benefit
        if self._is_hot_function(func):
            reasons.append("Function likely called frequently")
            return (JitStrategy.NJIT, 0.75, reasons)

        # Default: standard JIT as fallback
        reasons.append("Standard numerical function")
        return (JitStrategy.JIT, 0.65, reasons)

    def _has_fastmath_opportunities(self, func: FunctionDef) -> bool:
        """Check if function has operations benefiting from fastmath."""
        if func.body is None:
            return False

        checker = _FastmathOpportunityChecker()
        checker.visit(func.body)
        return checker.has_opportunities

    def _is_hot_function(self, func: FunctionDef) -> bool:
        """Heuristically check if function is likely called frequently."""
        # Functions with numeric-sounding names often are hot
        hot_keywords = {"compute", "calculate", "update", "step", "iterate", "kernel"}
        name_lower = func.name.lower()
        return any(kw in name_lower for kw in hot_keywords)

    def _determine_options(
        self,
        func: FunctionDef,
        strategy: JitStrategy,
        loop_analysis: list[LoopAnalysis],
        math_intensity: float,
    ) -> dict:
        """Determine optimal Numba options for the chosen strategy."""
        options = {
            "cache": True,  # Almost always beneficial
            "nopython": strategy.requires_nopython,
        }

        if strategy in {JitStrategy.NJIT_PARALLEL, JitStrategy.GUVECTORIZE}:
            options["parallel"] = True
            options["nogil"] = True  # Release GIL for parallel

        if strategy == JitStrategy.NJIT_FASTMATH:
            options["fastmath"] = True

        # Enable bounds checking for development (can be disabled in production)
        options["boundscheck"] = False

        return options

    def _add_warnings(
        self,
        decision: JitDecision,
        func: FunctionDef,
        memory_pattern: MemoryPattern,
    ) -> None:
        """Add relevant warnings to the decision."""
        if not memory_pattern.is_cache_friendly:
            decision.warnings.append("Memory access pattern may cause cache misses")
            decision.warnings.extend(memory_pattern.suggestions)

        if decision.strategy == JitStrategy.NJIT_FASTMATH:
            decision.warnings.append("fastmath may affect numerical precision")

        if decision.options.get("parallel"):
            decision.warnings.append("Parallel execution may not benefit small workloads")

    def _suggest_alternatives(
        self,
        primary: JitStrategy,
        loop_analysis: list[LoopAnalysis],
        vectorization: VectorizationInfo,
    ) -> list[JitStrategy]:
        """Suggest alternative strategies."""
        alternatives: list[JitStrategy] = []

        if primary == JitStrategy.NJIT_PARALLEL:
            alternatives.append(JitStrategy.NJIT)  # Non-parallel fallback

        if primary == JitStrategy.NJIT_FASTMATH:
            alternatives.append(JitStrategy.NJIT)  # Precise math fallback

        if vectorization.is_vectorizable and primary != JitStrategy.VECTORIZE:
            alternatives.append(JitStrategy.VECTORIZE)

        if primary != JitStrategy.JIT:
            alternatives.append(JitStrategy.JIT)  # Object mode fallback

        return alternatives

    def _detect_vectorizable_ops(self, func: FunctionDef) -> list[VectorizableOp]:
        """Find operations that can be vectorized."""
        return self._vectorization_analyzer.find_vectorizable_ops(func)


# =============================================================================
# Loop Optimizer
# =============================================================================


class LoopOptimizer:
    """
    Analyzes and optimizes loop structures for JIT compilation.

    This optimizer examines loops for:
    - prange eligibility (parallel iteration)
    - Loop fusion opportunities
    - Cache tiling benefits
    - Reduction pattern detection
    """

    def __init__(self) -> None:
        self._current_func_params: set[str] = set()

    def analyze_function_loops(self, func: FunctionDef) -> list[LoopAnalysis]:
        """Analyze all loops in a function for parallelization."""
        from mathviz.compiler.parallel_analyzer import ParallelAnalyzer

        analyzer = ParallelAnalyzer()
        results = analyzer.analyze_function(func)
        return [analysis for _, analysis in results]

    def optimize_loops(self, func: FunctionDef) -> list[LoopTransformation]:
        """
        Suggest loop transformations for better performance.

        Returns list of suggested transformations with descriptions.
        """
        transformations: list[LoopTransformation] = []

        if func.body is None:
            return transformations

        loops = self._collect_loops(func.body)

        for i, loop in enumerate(loops):
            # Check for prange
            if self._can_use_prange(loop):
                transformations.append(
                    LoopTransformation(
                        loop=loop,
                        transformation="prange",
                        description="Convert range() to prange() for parallel execution",
                        expected_benefit="Linear speedup with available cores",
                        prerequisites=["No loop-carried dependencies"],
                    )
                )

            # Check for reduction
            reduction = self._detect_reduction(loop)
            if reduction:
                transformations.append(
                    LoopTransformation(
                        loop=loop,
                        transformation="reduction",
                        description=f"Parallel reduction on '{reduction.variable}'",
                        expected_benefit="Parallel sum/product with tree reduction",
                        prerequisites=["Associative operation"],
                    )
                )

            # Check for tiling
            tiling = self._can_tile_loop(loop)
            if tiling:
                transformations.append(
                    LoopTransformation(
                        loop=loop,
                        transformation="tile",
                        description=f"Tile loop with block size {tiling.suggested_tile_size}",
                        expected_benefit=f"{tiling.estimated_cache_benefit:.0%} cache improvement",
                        prerequisites=["Large iteration count", "Regular access pattern"],
                    )
                )

            # Check for loop fusion with adjacent loop
            if i + 1 < len(loops) and self._can_fuse_loops(loops[i], loops[i + 1]):
                transformations.append(
                    LoopTransformation(
                        loop=loop,
                        transformation="fuse",
                        description="Fuse with following loop to improve locality",
                        expected_benefit="Reduced memory traffic",
                        prerequisites=["Same iteration space", "No dependencies"],
                    )
                )

        return transformations

    def _collect_loops(self, node: ASTNode) -> list[ForStatement]:
        """Collect all for loops in order of appearance."""
        collector = _LoopCollector()
        collector.visit(node)
        return collector.loops

    def _can_use_prange(self, loop: ForStatement) -> bool:
        """Check if loop can use Numba's prange."""
        from mathviz.compiler.parallel_analyzer import ParallelAnalyzer

        analyzer = ParallelAnalyzer()
        analysis = analyzer.analyze_loop(loop)
        return analysis.can_use_prange

    def _detect_reduction(self, loop: ForStatement) -> ReductionInfo | None:
        """Detect reduction patterns in a loop."""
        detector = _ReductionDetector(loop.variable)
        detector.visit(loop.body)

        if detector.reductions:
            # Return the first detected reduction
            var, op = detector.reductions[0]
            return ReductionInfo(
                variable=var,
                operator=op,
                is_parallel_safe=True,
            )
        return None

    def _can_fuse_loops(self, loop1: ForStatement, loop2: ForStatement) -> bool:
        """Check if two adjacent loops can be fused."""
        # Both must iterate over the same range
        if not self._same_iteration_space(loop1.iterable, loop2.iterable):
            return False

        # Check for dependencies between loops
        # (simplified check - real implementation would be more thorough)
        writes_1 = self._get_written_vars(loop1)
        reads_2 = self._get_read_vars(loop2)

        # If loop2 reads what loop1 writes (other than the loop variable),
        # fusion might change semantics
        return not (writes_1 & reads_2 - {loop1.variable})

    def _same_iteration_space(self, iter1: Expression, iter2: Expression) -> bool:
        """Check if two iterables represent the same iteration space."""
        # Simplified check - compare structure
        if type(iter1) != type(iter2):
            return False

        if isinstance(iter1, RangeExpression) and isinstance(iter2, RangeExpression):
            # Check if bounds are the same (structurally)
            return self._same_expr(iter1.start, iter2.start) and self._same_expr(
                iter1.end, iter2.end
            )

        if isinstance(iter1, CallExpression) and isinstance(iter2, CallExpression):
            if isinstance(iter1.callee, Identifier) and isinstance(iter2.callee, Identifier):
                if iter1.callee.name == iter2.callee.name == "range":
                    if len(iter1.arguments) == len(iter2.arguments):
                        return all(
                            self._same_expr(a1, a2)
                            for a1, a2 in zip(iter1.arguments, iter2.arguments, strict=False)
                        )

        return False

    def _same_expr(self, e1: Expression | None, e2: Expression | None) -> bool:
        """Check if two expressions are structurally the same."""
        if e1 is None and e2 is None:
            return True
        if e1 is None or e2 is None:
            return False
        if type(e1) != type(e2):
            return False

        if isinstance(e1, IntegerLiteral) and isinstance(e2, IntegerLiteral):
            return e1.value == e2.value
        if isinstance(e1, Identifier) and isinstance(e2, Identifier):
            return e1.name == e2.name

        return False

    def _get_written_vars(self, loop: ForStatement) -> set[str]:
        """Get variables written in a loop body."""
        collector = _WriteCollector()
        collector.visit(loop.body)
        return collector.written

    def _get_read_vars(self, loop: ForStatement) -> set[str]:
        """Get variables read in a loop body."""
        collector = _ReadCollector()
        collector.visit(loop.body)
        return collector.read

    def _can_tile_loop(self, loop: ForStatement) -> TilingInfo | None:
        """Check if loop benefits from cache tiling."""
        # Analyze array accesses in loop body
        access_analyzer = _ArrayAccessAnalyzer(loop.variable)
        access_analyzer.visit(loop.body)

        # Tiling benefits nested loops with 2D array access
        if access_analyzer.array_accesses < 2:
            return None

        # Estimate working set
        working_set = access_analyzer.estimated_working_set

        # If working set exceeds L1 cache, tiling might help
        if working_set > L1_CACHE_SIZE:
            # Suggest tile size based on L1 cache
            element_size = 8  # Assume 64-bit floats
            tile_size = int((L1_CACHE_SIZE / element_size) ** 0.5)
            tile_size = max(16, min(tile_size, 64))  # Clamp to reasonable range

            return TilingInfo(
                loop_variable=loop.variable,
                suggested_tile_size=tile_size,
                estimated_cache_benefit=min(0.5, working_set / L1_CACHE_SIZE / 4),
                data_reuse_factor=access_analyzer.data_reuse_factor,
            )

        return None


# =============================================================================
# Vectorization Analyzer
# =============================================================================


class VectorizationAnalyzer:
    """
    Analyzes functions for vectorization opportunities.

    Detects patterns suitable for:
    - NumPy vectorized operations
    - Numba @vectorize decorator
    - SIMD intrinsics
    """

    def analyze(self, func: FunctionDef) -> VectorizationInfo:
        """Analyze function for vectorization potential."""
        info = VectorizationInfo()

        if func.body is None:
            return info

        # Check if function is element-wise
        if self._is_element_wise_function(func):
            info.is_vectorizable = True
            info.recommended_strategy = "numba_vectorize"
            info.estimated_simd_speedup = 4.0  # Assume 4-wide SIMD

        # Find vectorizable operations
        info.vectorizable_ops = self.find_vectorizable_ops(func)

        # Check for blocking patterns
        info.blocking_reasons = self._find_blocking_patterns(func)
        if info.blocking_reasons:
            info.is_vectorizable = False

        return info

    def find_vectorizable_ops(self, func: FunctionDef) -> list[VectorizableOp]:
        """Find operations that can be vectorized."""
        ops: list[VectorizableOp] = []

        if func.body is None:
            return ops

        finder = _VectorizableOpFinder()
        finder.visit(func.body)
        return finder.ops

    def _is_element_wise_function(self, func: FunctionDef) -> bool:
        """Check if function performs purely element-wise operations."""
        if func.body is None:
            return False

        # Single return with expression
        if len(func.body.statements) != 1:
            return False

        stmt = func.body.statements[0]
        if not isinstance(stmt, ReturnStatement):
            return False

        if stmt.value is None:
            return False

        # Check if expression is element-wise
        return self._is_element_wise(stmt.value)

    def _is_element_wise(self, expr: Expression) -> bool:
        """Check if expression is element-wise (maps over inputs)."""
        if isinstance(expr, (Identifier, IntegerLiteral, FloatLiteral)):
            return True

        if isinstance(expr, BinaryExpression):
            # Arithmetic operations are element-wise
            if expr.operator in {
                BinaryOperator.ADD,
                BinaryOperator.SUB,
                BinaryOperator.MUL,
                BinaryOperator.DIV,
                BinaryOperator.POW,
                BinaryOperator.MOD,
            }:
                return self._is_element_wise(expr.left) and self._is_element_wise(expr.right)

        if isinstance(expr, UnaryExpression):
            return self._is_element_wise(expr.operand)

        if isinstance(expr, CallExpression) and isinstance(expr.callee, Identifier):
            # Math functions are element-wise
            if expr.callee.name in NUMBA_SUPPORTED_BUILTINS:
                return all(self._is_element_wise(arg) for arg in expr.arguments)

        if isinstance(expr, ConditionalExpression):
            return (
                self._is_element_wise(expr.condition)
                and self._is_element_wise(expr.then_expr)
                and self._is_element_wise(expr.else_expr)
            )

        return False

    def _find_blocking_patterns(self, func: FunctionDef) -> list[str]:
        """Find patterns that prevent vectorization."""
        blockers: list[str] = []

        if func.body is None:
            return blockers

        checker = _VectorizationBlockerChecker()
        checker.visit(func.body)
        return checker.blockers

    def _can_use_numpy_vectorize(self, func: FunctionDef) -> bool:
        """Check if function can use numpy.vectorize."""
        return self._is_element_wise_function(func)

    def _can_use_numba_vectorize(self, func: FunctionDef) -> bool:
        """Check if function can use @vectorize decorator."""
        # Numba vectorize requires scalar signature
        for param in func.parameters:
            if param.type_annotation and isinstance(param.type_annotation, GenericType):
                # Array parameter - can't use @vectorize directly
                return False

        return self._is_element_wise_function(func)


# =============================================================================
# Cache Optimizer
# =============================================================================


class CacheOptimizer:
    """
    Analyzes memory access patterns for cache optimization.

    Provides hints for:
    - Working set size estimation
    - Cache-unfriendly pattern detection
    - Data layout suggestions
    """

    def get_cache_hints(self, func: FunctionDef) -> CacheHints:
        """Analyze memory access patterns for cache optimization."""
        hints = CacheHints()

        if func.body is None:
            return hints

        # Estimate working set
        hints.working_set_size = self._estimate_working_set_size(func)

        # Check cache fit
        hints.fits_l1 = hints.working_set_size <= L1_CACHE_SIZE
        hints.fits_l2 = hints.working_set_size <= L2_CACHE_SIZE
        hints.fits_l3 = hints.working_set_size <= L3_CACHE_SIZE

        # Find cache-unfriendly patterns
        hints.cache_unfriendly_patterns = self._detect_cache_unfriendly_patterns(func)

        # Generate suggestions
        hints.suggestions = self._generate_suggestions(hints)

        return hints

    def _estimate_working_set_size(self, func: FunctionDef) -> int:
        """Estimate memory working set size in bytes."""
        if func.body is None:
            return 0

        estimator = _WorkingSetEstimator()
        estimator.visit(func.body)
        return estimator.estimate

    def _detect_cache_unfriendly_patterns(self, func: FunctionDef) -> list[str]:
        """Find patterns that cause cache misses."""
        patterns: list[str] = []

        if func.body is None:
            return patterns

        detector = _CachePatternDetector()
        detector.visit(func.body)
        return detector.unfriendly_patterns

    def _generate_suggestions(self, hints: CacheHints) -> list[str]:
        """Generate cache optimization suggestions."""
        suggestions: list[str] = []

        if not hints.fits_l1 and hints.fits_l2:
            suggestions.append("Consider loop tiling to improve L1 cache utilization")

        if not hints.fits_l2:
            suggestions.append("Working set exceeds L2 cache - minimize memory footprint")

        for pattern in hints.cache_unfriendly_patterns:
            if "column-major" in pattern.lower():
                suggestions.append("Use row-major iteration order for array access")
            if "random" in pattern.lower():
                suggestions.append("Sort data to improve spatial locality")

        return suggestions


# =============================================================================
# Cost Model
# =============================================================================


class CostModel:
    """
    Estimates execution costs and JIT compilation benefits.

    Uses heuristics to estimate:
    - Raw execution time based on operations
    - Speedup from JIT compilation
    - Whether JIT overhead is worth it
    """

    # Operation costs (relative to simple arithmetic)
    OP_COSTS: dict[str, float] = {
        "add": 1.0,
        "sub": 1.0,
        "mul": 1.0,
        "div": 4.0,
        "mod": 8.0,
        "pow": 20.0,
        "sqrt": 5.0,
        "sin": 15.0,
        "cos": 15.0,
        "tan": 20.0,
        "exp": 12.0,
        "log": 15.0,
        "memory_access": 50.0,  # L2/L3 miss
    }

    def estimate_execution_time(
        self,
        func: FunctionDef,
        input_sizes: dict[str, int] | None = None,
    ) -> float:
        """
        Estimate relative execution time.

        Returns a unitless cost metric (not actual time).
        """
        if func.body is None:
            return 0.0

        counter = _OperationCounter()
        counter.visit(func.body)

        # Base cost from operations
        cost = 0.0
        for op, count in counter.operation_counts.items():
            cost += self.OP_COSTS.get(op, 1.0) * count

        # Multiply by estimated iteration count if loops present
        if counter.loop_count > 0 and input_sizes:
            # Assume first numeric parameter determines size
            for param in func.parameters:
                if param.name in input_sizes:
                    cost *= input_sizes[param.name]
                    break
            else:
                cost *= 1000  # Default assumption

        return cost

    def estimate_jit_benefit(self, func: FunctionDef) -> float:
        """
        Estimate speedup factor from JIT compilation.

        Returns estimated speedup (e.g., 10.0 means 10x faster).
        """
        if func.body is None:
            return 1.0

        analyzer = _JitBenefitAnalyzer()
        analyzer.visit(func.body)

        # Base speedup factors
        speedup = 1.0

        # Loops benefit significantly
        if analyzer.has_loops:
            speedup *= 5.0

        # Math operations benefit from compilation
        if analyzer.math_op_count > 10:
            speedup *= 2.0

        # Array operations benefit
        if analyzer.array_access_count > 5:
            speedup *= 2.5

        # Simple functions get less benefit
        if analyzer.statement_count < 3 and not analyzer.has_loops:
            speedup *= 0.5

        # Cap at reasonable maximum
        return min(speedup, 50.0)

    def should_jit(self, func: FunctionDef) -> bool:
        """
        Decide if JIT compilation is worth the overhead.

        Returns True if expected benefit exceeds compilation cost.
        """
        benefit = self.estimate_jit_benefit(func)

        # JIT has upfront cost - need significant benefit
        return benefit >= 2.0


# =============================================================================
# Code Generation Integration
# =============================================================================


def generate_optimized_function(func: FunctionDef, decision: JitDecision) -> str:
    """
    Generate optimized Python code based on JIT decision.

    This function produces the decorator and any necessary transformations
    for the given JIT strategy.

    Args:
        func: The function definition AST node
        decision: The JIT optimization decision

    Returns:
        Generated Python code string for the function
    """
    lines: list[str] = []

    # Generate decorator based on strategy
    decorator = _generate_decorator(decision)
    if decorator:
        lines.append(decorator)

    # Generate function signature
    sig = _generate_function_signature(func)
    lines.append(sig)

    # Generate function body
    # (This would integrate with the full CodeGenerator)
    lines.append("    # Function body generated by CodeGenerator")
    lines.append("    pass")

    return "\n".join(lines)


def _generate_decorator(decision: JitDecision) -> str | None:
    """Generate the appropriate Numba decorator."""
    if decision.strategy == JitStrategy.NONE:
        return None

    opts = decision.options

    if decision.strategy == JitStrategy.VECTORIZE:
        # Vectorize needs type signatures
        return "@vectorize(cache=True)"

    if decision.strategy == JitStrategy.GUVECTORIZE:
        return "@guvectorize(cache=True)"

    if decision.strategy == JitStrategy.STENCIL:
        return "@stencil"

    # Build njit/jit options string
    opt_parts: list[str] = []

    if decision.strategy in {
        JitStrategy.NJIT,
        JitStrategy.NJIT_PARALLEL,
        JitStrategy.NJIT_FASTMATH,
    }:
        decorator_name = "njit"
    else:
        decorator_name = "jit"
        if opts.get("nopython", False):
            opt_parts.append("nopython=True")

    if opts.get("parallel", False):
        opt_parts.append("parallel=True")
    if opts.get("cache", True):
        opt_parts.append("cache=True")
    if opts.get("fastmath", False):
        opt_parts.append("fastmath=True")
    if opts.get("nogil", False):
        opt_parts.append("nogil=True")

    if opt_parts:
        return f"@{decorator_name}({', '.join(opt_parts)})"
    else:
        return f"@{decorator_name}"


def _generate_function_signature(func: FunctionDef) -> str:
    """Generate function signature line."""
    params: list[str] = []
    for param in func.parameters:
        if param.type_annotation:
            # Include type hints
            type_str = _type_to_python(param.type_annotation)
            params.append(f"{param.name}: {type_str}")
        else:
            params.append(param.name)

    param_str = ", ".join(params)

    if func.return_type:
        ret_type = _type_to_python(func.return_type)
        return f"def {func.name}({param_str}) -> {ret_type}:"
    else:
        return f"def {func.name}({param_str}):"


def _type_to_python(type_ann: TypeAnnotation) -> str:
    """Convert MathViz type annotation to Python type string."""
    if isinstance(type_ann, SimpleType):
        mapping = {
            "Int": "int",
            "Float": "float",
            "Bool": "bool",
            "String": "str",
            "None": "None",
            "Array": "np.ndarray",
            "Vec": "np.ndarray",
            "Mat": "np.ndarray",
        }
        return mapping.get(type_ann.name, type_ann.name)
    elif isinstance(type_ann, GenericType):
        base = type_ann.base
        args = ", ".join(_type_to_python(arg) for arg in type_ann.type_args)
        if base == "List":
            return f"list[{args}]"
        elif base == "Optional":
            return f"Optional[{args}]"
        return f"{base}[{args}]"
    return "Any"


# =============================================================================
# AST Visitor Helpers
# =============================================================================


class _NumbaCompatibilityVisitor(BaseASTVisitor):
    """Checks function body for Numba-incompatible constructs."""

    def __init__(self) -> None:
        self.blocking_reasons: list[str] = []

    def visit_call_expression(self, node: CallExpression) -> None:
        if isinstance(node.callee, Identifier):
            if node.callee.name in JIT_BLOCKING_FUNCTIONS:
                self.blocking_reasons.append(
                    f"Function call '{node.callee.name}' not supported in nopython mode"
                )
        super().visit_call_expression(node)

    def visit_string_literal(self, node) -> None:
        self.blocking_reasons.append("String operations not supported in nopython mode")

    def visit_set_literal(self, node) -> None:
        self.blocking_reasons.append("Set operations not supported in nopython mode")

    def visit_dict_literal(self, node) -> None:
        self.blocking_reasons.append("Dict operations not supported in nopython mode")


class _MemoryAccessAnalyzer(BaseASTVisitor):
    """Analyzes memory access patterns."""

    def __init__(self) -> None:
        self.arrays_accessed: set[str] = set()
        self.access_count = 0
        self.sequential_accesses = 0
        self.strided_accesses = 0
        self.random_accesses = 0

    def visit_index_expression(self, node: IndexExpression) -> None:
        if isinstance(node.object, Identifier):
            self.arrays_accessed.add(node.object.name)
            self.access_count += 1

            # Analyze index pattern
            if isinstance(node.index, Identifier):
                self.sequential_accesses += 1
            elif isinstance(node.index, BinaryExpression):
                self.strided_accesses += 1
            else:
                self.random_accesses += 1

        super().visit_index_expression(node)

    def get_pattern(self) -> MemoryPattern:
        if self.access_count == 0:
            return MemoryPattern(pattern=MemoryAccessPattern.SEQUENTIAL)

        if self.random_accesses > self.sequential_accesses:
            pattern = MemoryAccessPattern.RANDOM
            is_cache_friendly = False
        elif self.strided_accesses > self.sequential_accesses:
            pattern = MemoryAccessPattern.STRIDED
            is_cache_friendly = True
        else:
            pattern = MemoryAccessPattern.SEQUENTIAL
            is_cache_friendly = True

        return MemoryPattern(
            pattern=pattern,
            arrays_accessed=self.arrays_accessed,
            is_cache_friendly=is_cache_friendly,
        )


class _ArithmeticIntensityAnalyzer(BaseASTVisitor):
    """Calculates arithmetic intensity (ops per memory access)."""

    def __init__(self) -> None:
        self.arithmetic_ops = 0
        self.memory_accesses = 0

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        if node.operator in {
            BinaryOperator.ADD,
            BinaryOperator.SUB,
            BinaryOperator.MUL,
            BinaryOperator.DIV,
            BinaryOperator.MOD,
            BinaryOperator.POW,
        }:
            self.arithmetic_ops += 1
        super().visit_binary_expression(node)

    def visit_call_expression(self, node: CallExpression) -> None:
        if isinstance(node.callee, Identifier):
            if node.callee.name in NUMBA_SUPPORTED_BUILTINS:
                self.arithmetic_ops += 1
        super().visit_call_expression(node)

    def visit_index_expression(self, node: IndexExpression) -> None:
        self.memory_accesses += 1
        super().visit_index_expression(node)

    def get_intensity(self) -> float:
        if self.memory_accesses == 0:
            return float(self.arithmetic_ops) if self.arithmetic_ops else 0.0
        return self.arithmetic_ops / self.memory_accesses


class _FastmathOpportunityChecker(BaseASTVisitor):
    """Checks for operations that benefit from fastmath."""

    def __init__(self) -> None:
        self.has_opportunities = False

    def visit_call_expression(self, node: CallExpression) -> None:
        if isinstance(node.callee, Identifier):
            if node.callee.name in FASTMATH_BENEFICIAL_OPS:
                self.has_opportunities = True
        super().visit_call_expression(node)


class _LoopCollector(BaseASTVisitor):
    """Collects all for loops in order."""

    def __init__(self) -> None:
        self.loops: list[ForStatement] = []

    def visit_for_statement(self, node: ForStatement) -> None:
        self.loops.append(node)
        super().visit_for_statement(node)


class _ReductionDetector(BaseASTVisitor):
    """Detects reduction patterns in loops."""

    REDUCTION_OPS = {
        BinaryOperator.ADD: "sum",
        BinaryOperator.MUL: "product",
    }

    def __init__(self, loop_var: str) -> None:
        self.loop_var = loop_var
        self.reductions: list[tuple[str, str]] = []

    def visit_compound_assignment(self, node: CompoundAssignment) -> None:
        if isinstance(node.target, Identifier):
            var_name = node.target.name
            if var_name != self.loop_var and node.operator in self.REDUCTION_OPS:
                self.reductions.append((var_name, self.REDUCTION_OPS[node.operator]))


class _WriteCollector(BaseASTVisitor):
    """Collects variables written to."""

    def __init__(self) -> None:
        self.written: set[str] = set()

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        if isinstance(node.target, Identifier):
            self.written.add(node.target.name)
        super().visit_assignment_statement(node)

    def visit_compound_assignment(self, node: CompoundAssignment) -> None:
        if isinstance(node.target, Identifier):
            self.written.add(node.target.name)
        super().visit_compound_assignment(node)

    def visit_let_statement(self, node: LetStatement) -> None:
        self.written.add(node.name)
        super().visit_let_statement(node)


class _ReadCollector(BaseASTVisitor):
    """Collects variables read from."""

    def __init__(self) -> None:
        self.read: set[str] = set()

    def visit_identifier(self, node: Identifier) -> None:
        self.read.add(node.name)


class _ArrayAccessAnalyzer(BaseASTVisitor):
    """Analyzes array access patterns for tiling decisions."""

    def __init__(self, loop_var: str) -> None:
        self.loop_var = loop_var
        self.array_accesses = 0
        self.estimated_working_set = 0
        self.data_reuse_factor = 1.0

    def visit_index_expression(self, node: IndexExpression) -> None:
        self.array_accesses += 1
        # Estimate working set contribution
        self.estimated_working_set += 8 * 1000  # Assume 1000 elements, 8 bytes each
        super().visit_index_expression(node)


class _VectorizableOpFinder(BaseASTVisitor):
    """Finds vectorizable operations."""

    def __init__(self) -> None:
        self.ops: list[VectorizableOp] = []

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        if node.operator in {
            BinaryOperator.ADD,
            BinaryOperator.SUB,
            BinaryOperator.MUL,
            BinaryOperator.DIV,
        }:
            self.ops.append(
                VectorizableOp(
                    expression=node,
                    element_type="float64",
                    operation=node.operator.name.lower(),
                    estimated_simd_lanes=4,
                    location=node.location,
                )
            )
        super().visit_binary_expression(node)


class _VectorizationBlockerChecker(BaseASTVisitor):
    """Finds patterns that block vectorization."""

    def __init__(self) -> None:
        self.blockers: list[str] = []

    def visit_if_statement(self, node: IfStatement) -> None:
        # Data-dependent branches can block vectorization
        self.blockers.append("Conditional branch may prevent vectorization")
        super().visit_if_statement(node)

    def visit_while_statement(self, node: WhileStatement) -> None:
        self.blockers.append("While loop prevents vectorization")
        super().visit_while_statement(node)


class _WorkingSetEstimator(BaseASTVisitor):
    """Estimates working set size in bytes."""

    def __init__(self) -> None:
        self.estimate = 0
        self._seen_arrays: set[str] = set()

    def visit_index_expression(self, node: IndexExpression) -> None:
        if isinstance(node.object, Identifier):
            if node.object.name not in self._seen_arrays:
                self._seen_arrays.add(node.object.name)
                # Assume moderate array size
                self.estimate += 8 * 10000  # 10K elements, 8 bytes each
        super().visit_index_expression(node)


class _CachePatternDetector(BaseASTVisitor):
    """Detects cache-unfriendly access patterns."""

    def __init__(self) -> None:
        self.unfriendly_patterns: list[str] = []

    def visit_index_expression(self, node: IndexExpression) -> None:
        # Detect column-major access in row-major array
        if isinstance(node.object, IndexExpression):
            # Nested indexing like arr[j][i] - might be column-major
            self.unfriendly_patterns.append("Possible column-major access in row-major array")
        super().visit_index_expression(node)


class _OperationCounter(BaseASTVisitor):
    """Counts different types of operations."""

    def __init__(self) -> None:
        self.operation_counts: dict[str, int] = {}
        self.loop_count = 0

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        op = node.operator.name.lower()
        self.operation_counts[op] = self.operation_counts.get(op, 0) + 1
        super().visit_binary_expression(node)

    def visit_for_statement(self, node: ForStatement) -> None:
        self.loop_count += 1
        super().visit_for_statement(node)

    def visit_call_expression(self, node: CallExpression) -> None:
        if isinstance(node.callee, Identifier):
            func = node.callee.name.lower()
            self.operation_counts[func] = self.operation_counts.get(func, 0) + 1
        super().visit_call_expression(node)


class _JitBenefitAnalyzer(BaseASTVisitor):
    """Analyzes function to estimate JIT benefit."""

    def __init__(self) -> None:
        self.has_loops = False
        self.math_op_count = 0
        self.array_access_count = 0
        self.statement_count = 0

    def visit_for_statement(self, node: ForStatement) -> None:
        self.has_loops = True
        super().visit_for_statement(node)

    def visit_while_statement(self, node: WhileStatement) -> None:
        self.has_loops = True
        super().visit_while_statement(node)

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        self.math_op_count += 1
        super().visit_binary_expression(node)

    def visit_call_expression(self, node: CallExpression) -> None:
        if isinstance(node.callee, Identifier):
            if node.callee.name in NUMBA_SUPPORTED_BUILTINS:
                self.math_op_count += 1
        super().visit_call_expression(node)

    def visit_index_expression(self, node: IndexExpression) -> None:
        self.array_access_count += 1
        super().visit_index_expression(node)

    def visit_let_statement(self, node: LetStatement) -> None:
        self.statement_count += 1
        super().visit_let_statement(node)

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        self.statement_count += 1
        super().visit_assignment_statement(node)

    def visit_return_statement(self, node: ReturnStatement) -> None:
        self.statement_count += 1
        super().visit_return_statement(node)


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_jit_decision(
    func: FunctionDef,
    purity_info: dict[str, PurityInfo] | None = None,
    complexity_info: dict[str, ComplexityInfo] | None = None,
    parallel_info: dict[str, list[LoopAnalysis]] | None = None,
) -> JitDecision:
    """
    Convenience function to analyze a function for JIT optimization.

    Args:
        func: The function to analyze
        purity_info: Optional pre-computed purity analysis
        complexity_info: Optional pre-computed complexity analysis
        parallel_info: Optional pre-computed parallel analysis

    Returns:
        JitDecision with recommended strategy and rationale
    """
    analyzer = JitAnalyzer(
        purity_info=purity_info,
        complexity_info=complexity_info,
        parallel_info=parallel_info,
    )
    return analyzer.analyze_function(func)


def get_loop_optimizations(func: FunctionDef) -> list[LoopTransformation]:
    """
    Get suggested loop optimizations for a function.

    Args:
        func: The function to analyze

    Returns:
        List of suggested loop transformations
    """
    optimizer = LoopOptimizer()
    return optimizer.optimize_loops(func)


def estimate_jit_speedup(func: FunctionDef) -> float:
    """
    Estimate the speedup factor from JIT compilation.

    Args:
        func: The function to analyze

    Returns:
        Estimated speedup factor (e.g., 10.0 means 10x faster)
    """
    model = CostModel()
    return model.estimate_jit_benefit(func)


def is_numba_compatible(func: FunctionDef) -> tuple[bool, list[str]]:
    """
    Check if a function is compatible with Numba JIT.

    Args:
        func: The function to check

    Returns:
        Tuple of (is_compatible, list_of_blocking_reasons)
    """
    analyzer = JitAnalyzer()
    return analyzer._check_numba_compatibility(func)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Main classes
    "JitAnalyzer",
    "LoopOptimizer",
    "VectorizationAnalyzer",
    "CacheOptimizer",
    "CostModel",
    # Enums
    "JitStrategy",
    "MemoryAccessPattern",
    "LoopPattern",
    # Data classes
    "JitDecision",
    "VectorizableOp",
    "LoopTransformation",
    "ReductionInfo",
    "TilingInfo",
    "MemoryPattern",
    "VectorizationInfo",
    "CacheHints",
    # Convenience functions
    "analyze_jit_decision",
    "get_loop_optimizations",
    "estimate_jit_speedup",
    "is_numba_compatible",
    "generate_optimized_function",
    # Constants
    "NUMBA_COMPATIBLE_TYPES",
    "NUMBA_INCOMPATIBLE_TYPES",
    "NUMBA_SUPPORTED_BUILTINS",
    "JIT_BLOCKING_FUNCTIONS",
]
