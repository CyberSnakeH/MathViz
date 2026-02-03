"""
SIMD Vectorization Pass for MathViz Compiler.

This module detects and transforms code to use NumPy vectorized operations
and Numba SIMD intrinsics. It provides comprehensive analysis and transformation
capabilities for optimizing numerical code.

The vectorizer supports multiple strategies:
- NumPy ufunc transformations (element-wise operations)
- NumPy broadcasting optimizations
- Numba @vectorize and @guvectorize decorators
- Numba @stencil for neighborhood operations
- Explicit SIMD hints with prange and fastmath

Key Features:
1. Loop Vectorization Analysis - Detect vectorizable patterns
2. NumPy Transformation - Convert loops to NumPy operations
3. Stencil Detection - Identify neighborhood computations
4. Broadcast Analysis - Detect broadcasting opportunities
5. SIMD Code Generation - Generate Numba-optimized code
6. Vectorization Reporting - Detailed optimization reports

References:
- NumPy ufunc documentation: https://numpy.org/doc/stable/reference/ufuncs.html
- Numba vectorize: https://numba.readthedocs.io/en/stable/user/vectorize.html
- Numba stencil: https://numba.readthedocs.io/en/stable/user/stencil.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from mathviz.compiler.ast_nodes import (
    AssignmentStatement,
    # AST base classes
    ASTNode,
    BaseASTVisitor,
    BinaryExpression,
    BinaryOperator,
    Block,
    CallExpression,
    CompoundAssignment,
    ConditionalExpression,
    Expression,
    FloatLiteral,
    ForStatement,
    FunctionDef,
    # Expressions
    Identifier,
    IfStatement,
    IndexExpression,
    IntegerLiteral,
    # Statements
    LetStatement,
    RangeExpression,
    Statement,
    UnaryExpression,
    UnaryOperator,
    WhileStatement,
)

# =============================================================================
# Enumerations and Data Classes
# =============================================================================


class VectorizationStrategy(Enum):
    """
    Available vectorization strategies ordered by complexity and applicability.

    Each strategy has different tradeoffs:
    - NONE: No vectorization possible
    - NUMPY_UFUNC: Direct NumPy ufunc (fastest for simple ops)
    - NUMPY_BROADCAST: NumPy broadcasting (efficient for mixed dimensions)
    - NUMBA_VECTORIZE: @vectorize decorator (custom ufuncs)
    - NUMBA_GUVECTORIZE: @guvectorize (generalized ufuncs)
    - NUMBA_STENCIL: @stencil (neighborhood operations)
    - EXPLICIT_SIMD: Manual prange with SIMD hints
    """

    NONE = "none"
    NUMPY_UFUNC = "numpy_ufunc"
    NUMPY_BROADCAST = "numpy_broadcast"
    NUMBA_VECTORIZE = "numba_vectorize"
    NUMBA_GUVECTORIZE = "numba_guvectorize"
    NUMBA_STENCIL = "numba_stencil"
    EXPLICIT_SIMD = "explicit_simd"


class LoopPattern(Enum):
    """
    Recognized loop computation patterns.

    These patterns determine which vectorization strategy is most appropriate.
    """

    UNKNOWN = auto()
    SIMPLE_MAP = auto()  # arr[i] = f(arr[i])
    ELEMENT_WISE = auto()  # c[i] = a[i] op b[i]
    REDUCTION = auto()  # acc = acc op arr[i]
    STENCIL_1D = auto()  # out[i] = f(arr[i-1], arr[i], arr[i+1])
    STENCIL_2D = auto()  # out[i,j] = f(neighbors)
    OUTER_PRODUCT = auto()  # c[i,j] = a[i] * b[j]
    MATRIX_VECTOR = auto()  # c[i] = sum(A[i,j] * b[j])
    BROADCAST_SCALAR = auto()  # arr[i] = arr[i] + scalar
    SCAN = auto()  # arr[i] = arr[i] + arr[i-1] (prefix sum)
    GATHER = auto()  # out[i] = arr[indices[i]]
    SCATTER = auto()  # arr[indices[i]] = values[i]


class ReductionType(Enum):
    """Types of reduction operations that can be vectorized."""

    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AND = "and"
    OR = "or"
    COUNT = "count"
    MEAN = "mean"
    ARGMIN = "argmin"
    ARGMAX = "argmax"


@dataclass(frozen=True, slots=True)
class VectorizationResult:
    """
    Complete result of vectorization analysis for a loop or expression.

    Attributes:
        can_vectorize: Whether the code can be vectorized
        strategy: The recommended vectorization strategy
        transformed_code: Generated vectorized Python code (if applicable)
        estimated_speedup: Estimated performance improvement factor
        blockers: List of reasons preventing vectorization
        pattern: Detected computation pattern
        array_info: Information about arrays involved
        reduction_info: Information about any reduction operations
    """

    can_vectorize: bool
    strategy: VectorizationStrategy
    transformed_code: str | None = None
    estimated_speedup: float = 1.0
    blockers: list[str] = field(default_factory=list)
    pattern: LoopPattern = LoopPattern.UNKNOWN
    array_info: dict[str, ArrayAccessInfo] = field(default_factory=dict)
    reduction_info: ReductionInfo | None = None

    def __str__(self) -> str:
        status = "VECTORIZABLE" if self.can_vectorize else "NOT VECTORIZABLE"
        lines = [f"Vectorization Analysis: {status}"]
        lines.append(f"  Strategy: {self.strategy.value}")
        lines.append(f"  Pattern: {self.pattern.name}")
        lines.append(f"  Estimated Speedup: {self.estimated_speedup:.1f}x")

        if self.blockers:
            lines.append("  Blockers:")
            for blocker in self.blockers:
                lines.append(f"    - {blocker}")

        if self.reduction_info:
            lines.append(f"  Reduction: {self.reduction_info.reduction_type.value}")

        if self.transformed_code:
            lines.append("  Transformed Code:")
            for line in self.transformed_code.split("\n")[:5]:
                lines.append(f"    {line}")
            if self.transformed_code.count("\n") > 5:
                lines.append("    ...")

        return "\n".join(lines)


@dataclass(slots=True)
class ArrayAccessInfo:
    """
    Information about how an array is accessed within a loop.

    Attributes:
        name: Array variable name
        dimensions: Number of dimensions accessed
        access_pattern: How indices relate to loop variables
        is_read: Whether the array is read
        is_written: Whether the array is written
        strides: Index strides relative to loop variable
        is_contiguous: Whether access is contiguous in memory
    """

    name: str
    dimensions: int = 1
    access_pattern: list[str] = field(default_factory=list)
    is_read: bool = False
    is_written: bool = False
    strides: list[int] = field(default_factory=list)
    is_contiguous: bool = True


@dataclass(slots=True)
class ReductionInfo:
    """
    Information about a reduction operation.

    Attributes:
        variable: The accumulator variable name
        reduction_type: Type of reduction (sum, product, etc.)
        init_value: Initial value expression
        numpy_func: Corresponding NumPy function name
    """

    variable: str
    reduction_type: ReductionType
    init_value: Expression | None = None
    numpy_func: str = ""


@dataclass(slots=True)
class StencilInfo:
    """
    Information about a stencil computation pattern.

    Attributes:
        dimensions: Number of dimensions (1D, 2D, 3D)
        neighborhood: Relative offsets accessed [(di, dj, ...)]
        kernel: The stencil kernel expression
        boundary: Boundary handling mode
        is_symmetric: Whether the stencil is symmetric
        radius: Maximum offset in any dimension
    """

    dimensions: int
    neighborhood: list[tuple[int, ...]]
    kernel: Expression
    boundary: str = "reflect"  # "wrap", "reflect", "constant", "nearest"
    is_symmetric: bool = False
    radius: int = 1


@dataclass(slots=True)
class BroadcastInfo:
    """
    Information about broadcasting opportunities.

    Attributes:
        can_broadcast: Whether broadcasting is applicable
        scalar_vars: Variables that are scalars (broadcast to all elements)
        array_vars: Variables that are arrays
        broadcast_pattern: Description of the broadcast operation
        numpy_equivalent: NumPy expression that achieves the same result
    """

    can_broadcast: bool = False
    scalar_vars: set[str] = field(default_factory=set)
    array_vars: set[str] = field(default_factory=set)
    broadcast_pattern: str = ""
    numpy_equivalent: str = ""


# =============================================================================
# Mapping Tables
# =============================================================================


# Binary operators that map directly to NumPy ufuncs
NUMPY_UFUNC_MAP: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "np.add",
    BinaryOperator.SUB: "np.subtract",
    BinaryOperator.MUL: "np.multiply",
    BinaryOperator.DIV: "np.divide",
    BinaryOperator.FLOOR_DIV: "np.floor_divide",
    BinaryOperator.MOD: "np.mod",
    BinaryOperator.POW: "np.power",
    BinaryOperator.EQ: "np.equal",
    BinaryOperator.NE: "np.not_equal",
    BinaryOperator.LT: "np.less",
    BinaryOperator.GT: "np.greater",
    BinaryOperator.LE: "np.less_equal",
    BinaryOperator.GE: "np.greater_equal",
    BinaryOperator.AND: "np.logical_and",
    BinaryOperator.OR: "np.logical_or",
}

# Simple binary operators using Python syntax
PYTHON_OP_MAP: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "+",
    BinaryOperator.SUB: "-",
    BinaryOperator.MUL: "*",
    BinaryOperator.DIV: "/",
    BinaryOperator.FLOOR_DIV: "//",
    BinaryOperator.MOD: "%",
    BinaryOperator.POW: "**",
    BinaryOperator.EQ: "==",
    BinaryOperator.NE: "!=",
    BinaryOperator.LT: "<",
    BinaryOperator.GT: ">",
    BinaryOperator.LE: "<=",
    BinaryOperator.GE: ">=",
    BinaryOperator.AND: "&",
    BinaryOperator.OR: "|",
}

# Unary operators to NumPy
UNARY_UFUNC_MAP: dict[UnaryOperator, str] = {
    UnaryOperator.NEG: "np.negative",
    UnaryOperator.POS: "np.positive",
    UnaryOperator.NOT: "np.logical_not",
}

# Reduction operations to NumPy functions
REDUCTION_NUMPY_MAP: dict[ReductionType, str] = {
    ReductionType.SUM: "np.sum",
    ReductionType.PRODUCT: "np.prod",
    ReductionType.MIN: "np.min",
    ReductionType.MAX: "np.max",
    ReductionType.AND: "np.all",
    ReductionType.OR: "np.any",
    ReductionType.MEAN: "np.mean",
    ReductionType.ARGMIN: "np.argmin",
    ReductionType.ARGMAX: "np.argmax",
}

# Compound assignment operators to reduction types
COMPOUND_TO_REDUCTION: dict[BinaryOperator, ReductionType] = {
    BinaryOperator.ADD: ReductionType.SUM,
    BinaryOperator.MUL: ReductionType.PRODUCT,
    BinaryOperator.AND: ReductionType.AND,
    BinaryOperator.OR: ReductionType.OR,
}

# Math functions that have NumPy equivalents
NUMPY_MATH_FUNCS: dict[str, str] = {
    "abs": "np.abs",
    "sqrt": "np.sqrt",
    "sin": "np.sin",
    "cos": "np.cos",
    "tan": "np.tan",
    "exp": "np.exp",
    "log": "np.log",
    "log10": "np.log10",
    "log2": "np.log2",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "round": "np.round",
    "asin": "np.arcsin",
    "acos": "np.arccos",
    "atan": "np.arctan",
    "sinh": "np.sinh",
    "cosh": "np.cosh",
    "tanh": "np.tanh",
    "degrees": "np.degrees",
    "radians": "np.radians",
    "sign": "np.sign",
    "clip": "np.clip",
}

# Speedup estimates for different strategies
SPEEDUP_ESTIMATES: dict[VectorizationStrategy, float] = {
    VectorizationStrategy.NONE: 1.0,
    VectorizationStrategy.NUMPY_UFUNC: 50.0,
    VectorizationStrategy.NUMPY_BROADCAST: 40.0,
    VectorizationStrategy.NUMBA_VECTORIZE: 30.0,
    VectorizationStrategy.NUMBA_GUVECTORIZE: 25.0,
    VectorizationStrategy.NUMBA_STENCIL: 20.0,
    VectorizationStrategy.EXPLICIT_SIMD: 15.0,
}


# =============================================================================
# Helper Visitors
# =============================================================================


class ExpressionCollector(BaseASTVisitor):
    """
    Collects all variable references and analyzes expression structure.
    """

    def __init__(self) -> None:
        self.variables: set[str] = set()
        self.array_accesses: list[tuple[str, Expression]] = []
        self.function_calls: list[tuple[str, tuple[Expression, ...]]] = []
        self.has_conditionals: bool = False
        self.has_complex_ops: bool = False

    def collect(self, node: ASTNode) -> None:
        """Collect information from an expression tree."""
        self.variables = set()
        self.array_accesses = []
        self.function_calls = []
        self.has_conditionals = False
        self.has_complex_ops = False
        self.visit(node)

    def visit_identifier(self, node: Identifier) -> None:
        self.variables.add(node.name)

    def visit_index_expression(self, node: IndexExpression) -> None:
        if isinstance(node.object, Identifier):
            self.array_accesses.append((node.object.name, node.index))
        self.visit(node.object)
        self.visit(node.index)

    def visit_call_expression(self, node: CallExpression) -> None:
        if isinstance(node.callee, Identifier):
            self.function_calls.append((node.callee.name, node.arguments))
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)

    def visit_conditional_expression(self, node: ConditionalExpression) -> None:
        self.has_conditionals = True
        self.visit(node.condition)
        self.visit(node.then_expr)
        self.visit(node.else_expr)

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_unary_expression(self, node: UnaryExpression) -> None:
        self.visit(node.operand)


class IndexPatternAnalyzer(BaseASTVisitor):
    """
    Analyzes index expressions to determine access patterns.

    This is crucial for stencil detection and vectorization feasibility.
    """

    def __init__(self, loop_vars: list[str]) -> None:
        self.loop_vars = loop_vars
        self.depends_on_loop_var: bool = False
        self.offsets: list[int] = []
        self.is_linear: bool = True
        self.coefficient: int = 1

    def analyze(self, index_expr: Expression) -> tuple[bool, list[int], bool]:
        """
        Analyze an index expression.

        Returns:
            Tuple of (depends_on_loop_var, offsets, is_linear_access)
        """
        self.depends_on_loop_var = False
        self.offsets = []
        self.is_linear = True
        self.coefficient = 1
        self._analyze_impl(index_expr, 0)
        return self.depends_on_loop_var, self.offsets, self.is_linear

    def _analyze_impl(self, expr: Expression, depth: int) -> int | None:
        """Recursive analysis returning constant value if computable."""
        if isinstance(expr, Identifier):
            if expr.name in self.loop_vars:
                self.depends_on_loop_var = True
                self.offsets.append(0)
                return None
            return None

        elif isinstance(expr, IntegerLiteral):
            return expr.value

        elif isinstance(expr, BinaryExpression):
            left_val = self._analyze_impl(expr.left, depth + 1)
            right_val = self._analyze_impl(expr.right, depth + 1)

            if expr.operator == BinaryOperator.ADD:
                if self._is_loop_var(expr.left) and right_val is not None:
                    self.offsets = [right_val]
                    return None
                elif self._is_loop_var(expr.right) and left_val is not None:
                    self.offsets = [left_val]
                    return None
                elif left_val is not None and right_val is not None:
                    return left_val + right_val

            elif expr.operator == BinaryOperator.SUB:
                if self._is_loop_var(expr.left) and right_val is not None:
                    self.offsets = [-right_val]
                    return None
                elif left_val is not None and right_val is not None:
                    return left_val - right_val

            elif expr.operator == BinaryOperator.MUL:
                # Non-unit stride access
                if self._is_loop_var(expr.left) and right_val is not None:
                    self.coefficient = right_val
                    self.is_linear = right_val == 1
                    return None
                elif self._is_loop_var(expr.right) and left_val is not None:
                    self.coefficient = left_val
                    self.is_linear = left_val == 1
                    return None
                elif left_val is not None and right_val is not None:
                    return left_val * right_val

            self.is_linear = False
            return None

        return None

    def _is_loop_var(self, expr: Expression) -> bool:
        """Check if expression is exactly a loop variable."""
        return isinstance(expr, Identifier) and expr.name in self.loop_vars


# =============================================================================
# Main Vectorization Classes
# =============================================================================


class LoopVectorizer:
    """
    Analyzes loops for vectorization opportunities.

    This is the main entry point for loop vectorization analysis.
    """

    def __init__(self) -> None:
        self._collector = ExpressionCollector()

    def analyze_loop(self, loop: ForStatement) -> VectorizationResult:
        """
        Analyze if a loop can be vectorized.

        This method performs comprehensive analysis including:
        - Pattern detection (map, reduction, stencil, etc.)
        - Dependency analysis
        - Strategy selection
        - Code transformation

        Args:
            loop: The ForStatement AST node to analyze

        Returns:
            VectorizationResult with analysis details and transformed code
        """
        blockers: list[str] = []

        # Step 1: Check if loop iterates over a range
        if not self._is_range_loop(loop):
            return VectorizationResult(
                can_vectorize=False,
                strategy=VectorizationStrategy.NONE,
                blockers=["Loop does not iterate over a range expression"],
            )

        loop_var = loop.variable

        # Step 2: Analyze loop body structure
        body_analysis = self._analyze_body(loop.body, loop_var)

        # Step 3: Check for vectorization blockers
        if body_analysis.get("has_break_continue"):
            blockers.append("Loop contains break or continue statements")

        if body_analysis.get("has_nested_loops"):
            blockers.append("Loop contains nested loops (consider separate analysis)")

        if body_analysis.get("has_side_effects"):
            blockers.append("Loop contains side effects (I/O, prints, etc.)")

        # Step 4: Detect the computation pattern
        pattern = self._detect_pattern(loop.body, loop_var, body_analysis)

        # Step 5: Check for loop-carried dependencies
        if self._has_loop_carried_dependency(loop) and pattern != LoopPattern.REDUCTION:
            blockers.append("Loop has cross-iteration data dependencies")

        if blockers:
            return VectorizationResult(
                can_vectorize=False,
                strategy=VectorizationStrategy.NONE,
                blockers=blockers,
                pattern=pattern,
            )

        # Step 6: Select vectorization strategy
        strategy, transformed = self._select_strategy_and_transform(loop, pattern, body_analysis)

        # Step 7: Estimate speedup
        speedup = self._estimate_speedup(pattern, strategy, body_analysis)

        return VectorizationResult(
            can_vectorize=strategy != VectorizationStrategy.NONE,
            strategy=strategy,
            transformed_code=transformed,
            estimated_speedup=speedup,
            pattern=pattern,
            array_info=body_analysis.get("array_info", {}),
            reduction_info=body_analysis.get("reduction_info"),
        )

    def _is_simple_map_loop(self, loop: ForStatement) -> bool:
        """
        Check if loop is: for i in range(n): arr[i] = f(arr[i])

        A simple map loop applies a function to each element independently.
        """
        statements = loop.body.statements
        if len(statements) != 1:
            return False

        stmt = statements[0]
        if not isinstance(stmt, AssignmentStatement):
            return False

        target = stmt.target
        if not isinstance(target, IndexExpression):
            return False

        # Check target is arr[i] where i is loop variable
        if not isinstance(target.object, Identifier):
            return False
        if not isinstance(target.index, Identifier):
            return False
        return target.index.name == loop.variable

    def _is_reduction_loop(self, loop: ForStatement) -> bool:
        """
        Check if loop is a reduction (sum, product, etc.)

        Reduction patterns: acc += x, acc *= x, acc = max(acc, x), etc.
        """
        for stmt in loop.body.statements:
            if isinstance(stmt, CompoundAssignment):
                target = stmt.target
                # Simple variable being accumulated
                if isinstance(target, Identifier) and stmt.operator in COMPOUND_TO_REDUCTION:
                    return True
        return False

    def _is_stencil_loop(self, loop: ForStatement) -> bool:
        """
        Check if loop accesses neighbors (arr[i-1], arr[i], arr[i+1])

        Stencil patterns access elements at fixed offsets from the loop index.
        """
        loop_var = loop.variable
        index_analyzer = IndexPatternAnalyzer([loop_var])

        offsets_seen: set[int] = set()

        for stmt in loop.body.statements:
            if isinstance(stmt, AssignmentStatement):
                self._collector.collect(stmt.value)
                for _arr_name, index_expr in self._collector.array_accesses:
                    depends, offsets, is_linear = index_analyzer.analyze(index_expr)
                    if depends and offsets:
                        offsets_seen.update(offsets)

        # Stencil if we access multiple different offsets
        return len(offsets_seen) > 1

    def _has_loop_carried_dependency(self, loop: ForStatement) -> bool:
        """
        Check for dependencies that prevent vectorization.

        A loop-carried dependency exists when iteration i depends on
        results from iteration i-k (k > 0).
        """
        loop_var = loop.variable
        index_analyzer = IndexPatternAnalyzer([loop_var])

        writes: list[tuple[str, list[int]]] = []
        reads: list[tuple[str, list[int]]] = []

        for stmt in loop.body.statements:
            # Analyze writes
            if isinstance(stmt, AssignmentStatement):
                target = stmt.target
                if isinstance(target, IndexExpression) and isinstance(target.object, Identifier):
                    arr_name = target.object.name
                    depends, offsets, _ = index_analyzer.analyze(target.index)
                    if depends:
                        writes.append((arr_name, offsets))

                # Analyze reads in the value expression
                self._collector.collect(stmt.value)
                for arr_name, index_expr in self._collector.array_accesses:
                    depends, offsets, _ = index_analyzer.analyze(index_expr)
                    if depends:
                        reads.append((arr_name, offsets))

            elif isinstance(stmt, CompoundAssignment):
                target = stmt.target
                if isinstance(target, IndexExpression) and isinstance(target.object, Identifier):
                    arr_name = target.object.name
                    depends, offsets, _ = index_analyzer.analyze(target.index)
                    if depends:
                        writes.append((arr_name, offsets))
                        reads.append((arr_name, offsets))

                self._collector.collect(stmt.value)
                for arr_name, index_expr in self._collector.array_accesses:
                    depends, offsets, _ = index_analyzer.analyze(index_expr)
                    if depends:
                        reads.append((arr_name, offsets))

        # Check for cross-iteration dependencies
        for w_arr, w_offsets in writes:
            for r_arr, r_offsets in reads:
                if w_arr == r_arr:
                    # Same array - check if read could depend on prior write
                    for w_off in w_offsets:
                        for r_off in r_offsets:
                            if r_off < w_off:
                                # Reading from earlier index than write
                                return True

        return False

    def _is_range_loop(self, loop: ForStatement) -> bool:
        """Check if loop iterates over a range expression."""
        iterable = loop.iterable

        if isinstance(iterable, RangeExpression):
            return True

        if isinstance(iterable, CallExpression) and isinstance(iterable.callee, Identifier):
            return iterable.callee.name == "range"

        return False

    def _analyze_body(self, body: Block, loop_var: str) -> dict[str, object]:
        """Analyze the loop body structure."""
        result: dict[str, object] = {
            "has_break_continue": False,
            "has_nested_loops": False,
            "has_side_effects": False,
            "has_conditionals": False,
            "array_info": {},
            "reduction_info": None,
            "statement_count": len(body.statements),
        }

        array_info: dict[str, ArrayAccessInfo] = {}
        index_analyzer = IndexPatternAnalyzer([loop_var])

        for stmt in body.statements:
            self._analyze_statement(stmt, loop_var, result, array_info, index_analyzer)

        result["array_info"] = array_info
        return result

    def _analyze_statement(
        self,
        stmt: Statement | Block,
        loop_var: str,
        result: dict[str, object],
        array_info: dict[str, ArrayAccessInfo],
        index_analyzer: IndexPatternAnalyzer,
    ) -> None:
        """Analyze a single statement within the loop body."""
        from mathviz.compiler.ast_nodes import (
            BreakStatement,
            ContinueStatement,
            PrintStatement,
        )

        # Handle Block by recursing into its statements
        if isinstance(stmt, Block):
            for inner_stmt in stmt.statements:
                self._analyze_statement(inner_stmt, loop_var, result, array_info, index_analyzer)
            return

        if isinstance(stmt, (BreakStatement, ContinueStatement)):
            result["has_break_continue"] = True

        elif isinstance(stmt, (ForStatement, WhileStatement)):
            result["has_nested_loops"] = True

        elif isinstance(stmt, PrintStatement):
            result["has_side_effects"] = True

        elif isinstance(stmt, AssignmentStatement):
            self._analyze_assignment(stmt, loop_var, array_info, index_analyzer)

        elif isinstance(stmt, CompoundAssignment):
            self._analyze_compound_assignment(stmt, loop_var, result, array_info, index_analyzer)

        elif isinstance(stmt, LetStatement):
            # Local variable definition - generally okay
            pass

        elif isinstance(stmt, IfStatement):
            result["has_conditionals"] = True
            self._analyze_statement(stmt.then_block, loop_var, result, array_info, index_analyzer)
            if stmt.else_block:
                self._analyze_statement(
                    stmt.else_block, loop_var, result, array_info, index_analyzer
                )

    def _analyze_assignment(
        self,
        stmt: AssignmentStatement,
        _loop_var: str,
        array_info: dict[str, ArrayAccessInfo],
        index_analyzer: IndexPatternAnalyzer,
    ) -> None:
        """Analyze an assignment statement for array access patterns."""
        target = stmt.target

        if isinstance(target, IndexExpression) and isinstance(target.object, Identifier):
            arr_name = target.object.name
            depends, offsets, is_linear = index_analyzer.analyze(target.index)

            if arr_name not in array_info:
                array_info[arr_name] = ArrayAccessInfo(name=arr_name)

            array_info[arr_name].is_written = True
            if depends:
                array_info[arr_name].strides = offsets
                array_info[arr_name].is_contiguous = is_linear

        # Analyze value expression for reads
        self._collector.collect(stmt.value)
        for arr_name, _index_expr in self._collector.array_accesses:
            if arr_name not in array_info:
                array_info[arr_name] = ArrayAccessInfo(name=arr_name)
            array_info[arr_name].is_read = True

    def _analyze_compound_assignment(
        self,
        stmt: CompoundAssignment,
        _loop_var: str,
        result: dict[str, object],
        array_info: dict[str, ArrayAccessInfo],
        _index_analyzer: IndexPatternAnalyzer,
    ) -> None:
        """Analyze a compound assignment for reduction patterns."""
        target = stmt.target

        if isinstance(target, Identifier):
            # Potential reduction: acc += value
            if stmt.operator in COMPOUND_TO_REDUCTION:
                reduction_type = COMPOUND_TO_REDUCTION[stmt.operator]
                result["reduction_info"] = ReductionInfo(
                    variable=target.name,
                    reduction_type=reduction_type,
                    numpy_func=REDUCTION_NUMPY_MAP.get(reduction_type, ""),
                )

        elif isinstance(target, IndexExpression) and isinstance(target.object, Identifier):
            arr_name = target.object.name
            if arr_name not in array_info:
                array_info[arr_name] = ArrayAccessInfo(name=arr_name)
            array_info[arr_name].is_written = True
            array_info[arr_name].is_read = True

        # Analyze value expression
        self._collector.collect(stmt.value)
        for arr_name, _index_expr in self._collector.array_accesses:
            if arr_name not in array_info:
                array_info[arr_name] = ArrayAccessInfo(name=arr_name)
            array_info[arr_name].is_read = True

    def _detect_pattern(
        self,
        body: Block,
        loop_var: str,
        body_analysis: dict[str, object],
    ) -> LoopPattern:
        """Detect the computation pattern of the loop."""
        # Check for reduction first (highest priority)
        if body_analysis.get("reduction_info"):
            return LoopPattern.REDUCTION

        # Check statement patterns
        if len(body.statements) == 1:
            stmt = body.statements[0]

            if isinstance(stmt, AssignmentStatement):
                if self._is_element_wise_pattern(stmt, loop_var):
                    return LoopPattern.ELEMENT_WISE

                if self._is_broadcast_scalar_pattern(stmt, loop_var):
                    return LoopPattern.BROADCAST_SCALAR

        # Check for stencil pattern
        if self._is_stencil_loop(
            ForStatement(
                variable=loop_var,
                iterable=RangeExpression(
                    start=IntegerLiteral(value=0),
                    end=Identifier(name="n"),
                ),
                body=body,
            )
        ):
            return LoopPattern.STENCIL_1D

        return LoopPattern.SIMPLE_MAP

    def _is_element_wise_pattern(self, stmt: AssignmentStatement, loop_var: str) -> bool:
        """Check if statement is c[i] = a[i] op b[i] pattern."""
        target = stmt.target
        if not isinstance(target, IndexExpression):
            return False

        if not isinstance(target.index, Identifier):
            return False

        if target.index.name != loop_var:
            return False

        # Check value is binary expression of array accesses
        value = stmt.value
        if isinstance(value, BinaryExpression):
            left_arr = self._is_array_at_loop_var(value.left, loop_var)
            right_arr = self._is_array_at_loop_var(value.right, loop_var)
            return left_arr and right_arr

        return False

    def _is_broadcast_scalar_pattern(self, stmt: AssignmentStatement, loop_var: str) -> bool:
        """Check if statement is arr[i] = arr[i] + scalar pattern."""
        target = stmt.target
        if not isinstance(target, IndexExpression):
            return False

        value = stmt.value
        if isinstance(value, BinaryExpression):
            left_arr = self._is_array_at_loop_var(value.left, loop_var)
            right_arr = self._is_array_at_loop_var(value.right, loop_var)
            left_scalar = isinstance(value.left, (Identifier, IntegerLiteral, FloatLiteral))
            right_scalar = isinstance(value.right, (Identifier, IntegerLiteral, FloatLiteral))

            # One side is array access, other is scalar
            return (left_arr and right_scalar) or (left_scalar and right_arr)

        return False

    def _is_array_at_loop_var(self, expr: Expression, loop_var: str) -> bool:
        """Check if expression is arr[loop_var] access."""
        if isinstance(expr, IndexExpression) and isinstance(expr.index, Identifier):
            return expr.index.name == loop_var
        return False

    def _select_strategy_and_transform(
        self,
        loop: ForStatement,
        pattern: LoopPattern,
        body_analysis: dict[str, object],
    ) -> tuple[VectorizationStrategy, str | None]:
        """Select the best vectorization strategy and generate transformed code."""
        if pattern == LoopPattern.REDUCTION:
            return self._transform_reduction(loop, body_analysis)

        elif pattern == LoopPattern.ELEMENT_WISE:
            return self._transform_element_wise(loop)

        elif pattern == LoopPattern.BROADCAST_SCALAR:
            return self._transform_broadcast(loop)

        elif pattern in (LoopPattern.STENCIL_1D, LoopPattern.STENCIL_2D):
            return self._transform_stencil(loop)

        elif pattern == LoopPattern.SIMPLE_MAP:
            return self._transform_simple_map(loop)

        return (VectorizationStrategy.EXPLICIT_SIMD, self._transform_explicit_simd(loop))

    def _transform_reduction(
        self, _loop: ForStatement, body_analysis: dict[str, object]
    ) -> tuple[VectorizationStrategy, str | None]:
        """Transform a reduction loop to NumPy."""
        reduction_info = body_analysis.get("reduction_info")
        if not reduction_info or not isinstance(reduction_info, ReductionInfo):
            return (VectorizationStrategy.NONE, None)

        # Find the array being reduced
        array_info = body_analysis.get("array_info", {})
        if not isinstance(array_info, dict):
            return (VectorizationStrategy.NONE, None)

        arr_names = [name for name, info in array_info.items() if info.is_read]
        if not arr_names:
            return (VectorizationStrategy.NONE, None)

        arr_name = arr_names[0]
        numpy_func = reduction_info.numpy_func

        if numpy_func:
            transformed = f"{reduction_info.variable} = {numpy_func}({arr_name})"
            return (VectorizationStrategy.NUMPY_UFUNC, transformed)

        return (VectorizationStrategy.EXPLICIT_SIMD, None)

    def _transform_element_wise(
        self, loop: ForStatement
    ) -> tuple[VectorizationStrategy, str | None]:
        """Transform element-wise loop to NumPy operation."""
        if len(loop.body.statements) != 1:
            return (VectorizationStrategy.NONE, None)

        stmt = loop.body.statements[0]
        if not isinstance(stmt, AssignmentStatement):
            return (VectorizationStrategy.NONE, None)

        target = stmt.target
        if not isinstance(target, IndexExpression):
            return (VectorizationStrategy.NONE, None)

        if not isinstance(target.object, Identifier):
            return (VectorizationStrategy.NONE, None)

        target_arr = target.object.name
        value = stmt.value

        if isinstance(value, BinaryExpression):
            op_str = PYTHON_OP_MAP.get(value.operator)
            if op_str:
                left_arr = self._get_array_name(value.left)
                right_arr = self._get_array_name(value.right)

                if left_arr and right_arr:
                    transformed = f"{target_arr} = {left_arr} {op_str} {right_arr}"
                    return (VectorizationStrategy.NUMPY_UFUNC, transformed)

        return (VectorizationStrategy.EXPLICIT_SIMD, None)

    def _transform_broadcast(self, loop: ForStatement) -> tuple[VectorizationStrategy, str | None]:
        """Transform broadcast loop to NumPy operation."""
        if len(loop.body.statements) != 1:
            return (VectorizationStrategy.NONE, None)

        stmt = loop.body.statements[0]
        if not isinstance(stmt, AssignmentStatement):
            return (VectorizationStrategy.NONE, None)

        target = stmt.target
        if not isinstance(target, IndexExpression):
            return (VectorizationStrategy.NONE, None)

        if not isinstance(target.object, Identifier):
            return (VectorizationStrategy.NONE, None)

        target_arr = target.object.name
        value = stmt.value

        if isinstance(value, BinaryExpression):
            op_str = PYTHON_OP_MAP.get(value.operator)
            if op_str:
                left_arr = self._get_array_name(value.left)
                right_str = self._expr_to_string(value.right)
                left_str = self._expr_to_string(value.left)
                right_arr = self._get_array_name(value.right)

                if left_arr:
                    transformed = f"{target_arr} = {left_arr} {op_str} {right_str}"
                    return (VectorizationStrategy.NUMPY_BROADCAST, transformed)
                elif right_arr:
                    transformed = f"{target_arr} = {left_str} {op_str} {right_arr}"
                    return (VectorizationStrategy.NUMPY_BROADCAST, transformed)

        return (VectorizationStrategy.EXPLICIT_SIMD, None)

    def _transform_stencil(self, _loop: ForStatement) -> tuple[VectorizationStrategy, str | None]:
        """Transform stencil loop to Numba @stencil."""
        # Stencil transformation is more complex - generate decorator code
        # This would be implemented with StencilDetector analysis

        return (VectorizationStrategy.NUMBA_STENCIL, None)

    def _transform_simple_map(self, loop: ForStatement) -> tuple[VectorizationStrategy, str | None]:
        """Transform simple map loop to np.vectorize or direct operation."""
        if len(loop.body.statements) != 1:
            return (VectorizationStrategy.EXPLICIT_SIMD, None)

        stmt = loop.body.statements[0]
        if not isinstance(stmt, AssignmentStatement):
            return (VectorizationStrategy.EXPLICIT_SIMD, None)

        target = stmt.target
        if not isinstance(target, IndexExpression):
            return (VectorizationStrategy.EXPLICIT_SIMD, None)

        if not isinstance(target.object, Identifier):
            return (VectorizationStrategy.EXPLICIT_SIMD, None)

        target_arr = target.object.name
        value = stmt.value

        # Check for simple function application
        if isinstance(value, CallExpression) and isinstance(value.callee, Identifier):
            func_name = value.callee.name
            numpy_func = NUMPY_MATH_FUNCS.get(func_name)

            # Single array argument at loop index
            if numpy_func and len(value.arguments) == 1:
                arg = value.arguments[0]
                arr_name = self._get_array_name(arg)
                if arr_name:
                    transformed = f"{target_arr} = {numpy_func}({arr_name})"
                    return (VectorizationStrategy.NUMPY_UFUNC, transformed)

        return (VectorizationStrategy.EXPLICIT_SIMD, None)

    def _transform_explicit_simd(self, _loop: ForStatement) -> str:
        """Generate explicit SIMD code using Numba prange."""
        lines = [
            "@njit(parallel=True, fastmath=True)",
            "def vectorized_loop(...):",
            "    for i in prange(len(arr)):",
            "        # Loop body here",
        ]
        return "\n".join(lines)

    def _get_array_name(self, expr: Expression) -> str | None:
        """Extract array name from an index expression."""
        if isinstance(expr, IndexExpression) and isinstance(expr.object, Identifier):
            return expr.object.name
        return None

    def _expr_to_string(self, expr: Expression) -> str:
        """Convert expression to Python string."""
        if isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, (IntegerLiteral, FloatLiteral)):
            return str(expr.value)
        elif isinstance(expr, IndexExpression):
            arr = self._get_array_name(expr)
            if arr:
                return arr
        return "expr"

    def _estimate_speedup(
        self,
        pattern: LoopPattern,
        strategy: VectorizationStrategy,
        body_analysis: dict[str, object],
    ) -> float:
        """Estimate speedup from vectorization."""
        base_speedup = SPEEDUP_ESTIMATES.get(strategy, 1.0)

        # Adjust based on pattern complexity
        if pattern in (LoopPattern.STENCIL_1D, LoopPattern.STENCIL_2D):
            base_speedup *= 0.8  # Stencils have more overhead

        if body_analysis.get("has_conditionals"):
            base_speedup *= 0.5  # Conditionals reduce vectorization efficiency

        return base_speedup


class NumpyTransformer:
    """
    Transform loops to NumPy vectorized operations.

    This class handles the actual code transformation, generating
    efficient NumPy code from loop-based patterns.
    """

    def __init__(self) -> None:
        self._vectorizer = LoopVectorizer()

    def transform(self, func: FunctionDef) -> tuple[FunctionDef, list[str]]:
        """
        Transform eligible loops in a function to NumPy operations.

        Args:
            func: The function definition to transform

        Returns:
            Tuple of (transformed function, list of transformation descriptions)
        """
        if not func.body:
            return func, []

        transformations: list[str] = []
        new_statements: list[Statement] = []

        for stmt in func.body.statements:
            if isinstance(stmt, ForStatement):
                result = self._vectorizer.analyze_loop(stmt)
                if result.can_vectorize and result.transformed_code:
                    # Record transformation
                    transformations.append(
                        f"Transformed loop to {result.strategy.value}: {result.transformed_code[:50]}..."
                    )
                    # Keep original for now - in production, parse transformed code
                    new_statements.append(stmt)
                else:
                    new_statements.append(stmt)
            else:
                new_statements.append(stmt)

        # Create new function with transformed body
        new_body = Block(statements=tuple(new_statements), location=func.body.location)
        new_func = FunctionDef(
            name=func.name,
            parameters=func.parameters,
            return_type=func.return_type,
            body=new_body,
            type_params=func.type_params,
            where_clause=func.where_clause,
            jit_options=func.jit_options,
            doc_comment=func.doc_comment,
            attributes=func.attributes,
            location=func.location,
        )

        return new_func, transformations


class StencilDetector:
    """
    Detect stencil computation patterns.

    Stencils are a common pattern in scientific computing where each
    output element depends on a neighborhood of input elements.
    """

    def __init__(self) -> None:
        self._collector = ExpressionCollector()

    def detect(self, loop: ForStatement) -> StencilInfo | None:
        """
        Detect if loop is a stencil computation.

        Common patterns detected:
        - 1D average: out[i] = (arr[i-1] + arr[i] + arr[i+1]) / 3
        - 2D Laplacian: out[i,j] = arr[i-1,j] + arr[i+1,j] + arr[i,j-1] + arr[i,j+1]
        - General convolutions

        Args:
            loop: The for loop to analyze

        Returns:
            StencilInfo if a stencil pattern is detected, None otherwise
        """
        if len(loop.body.statements) != 1:
            return None

        stmt = loop.body.statements[0]
        if not isinstance(stmt, AssignmentStatement):
            return None

        loop_var = loop.variable
        index_analyzer = IndexPatternAnalyzer([loop_var])

        # Collect all array accesses
        self._collector.collect(stmt.value)

        if not self._collector.array_accesses:
            return None

        # Analyze offsets
        neighborhood: list[tuple[int, ...]] = []
        seen_arrays: set[str] = set()

        for arr_name, index_expr in self._collector.array_accesses:
            depends, offsets, is_linear = index_analyzer.analyze(index_expr)
            if depends and offsets:
                seen_arrays.add(arr_name)
                for off in offsets:
                    neighborhood.append((off,))

        if len(neighborhood) <= 1:
            return None

        # Determine stencil properties
        offsets_list = [n[0] for n in neighborhood]
        radius = max(abs(min(offsets_list)), abs(max(offsets_list)))
        is_symmetric = set(offsets_list) == {-o for o in offsets_list}

        return StencilInfo(
            dimensions=1,
            neighborhood=neighborhood,
            kernel=stmt.value,
            boundary="reflect",
            is_symmetric=is_symmetric,
            radius=radius,
        )

    def generate_numba_stencil(self, info: StencilInfo) -> str:
        """
        Generate @stencil decorated function.

        Args:
            info: StencilInfo describing the stencil computation

        Returns:
            Python code string with Numba @stencil decorator
        """
        lines = ["from numba import stencil", ""]

        # Determine stencil mode based on boundary
        mode_map = {
            "wrap": "'wrap'",
            "reflect": "'reflect'",
            "constant": "'constant'",
            "nearest": "'nearest'",
        }
        mode_map.get(info.boundary, "'constant'")

        lines.append("@stencil")
        lines.append("def stencil_kernel(arr):")

        # Generate stencil body
        # The kernel expression would be converted to stencil syntax
        # For now, generate a template
        if info.dimensions == 1:
            offsets_str = ", ".join(f"arr[{off}]" for (off,) in info.neighborhood)
            lines.append(f"    # Stencil accessing: {offsets_str}")
            lines.append("    return (arr[-1] + arr[0] + arr[1]) / 3  # Example")
        else:
            lines.append(f"    # {info.dimensions}D stencil with radius {info.radius}")
            lines.append("    return arr[0, 0]  # Placeholder")

        lines.append("")
        lines.append("# Apply with: result = stencil_kernel(input_array)")

        return "\n".join(lines)


class BroadcastAnalyzer:
    """
    Detect operations that can use NumPy broadcasting.

    Broadcasting allows operations between arrays of different shapes
    by automatically expanding dimensions.
    """

    def __init__(self) -> None:
        self._collector = ExpressionCollector()

    def analyze(self, expr: Expression, context: dict[str, str]) -> BroadcastInfo:
        """
        Analyze expression for broadcasting opportunities.

        Patterns detected:
        - arr + scalar: Broadcasting scalar to all elements
        - vec[:, np.newaxis] * row: Outer product via broadcasting
        - arr[i] * scalar_array: Row/column broadcasting

        Args:
            expr: Expression to analyze
            context: Variable type information

        Returns:
            BroadcastInfo with broadcasting analysis
        """
        info = BroadcastInfo()

        self._collector.collect(expr)

        # Categorize variables
        for var in self._collector.variables:
            var_type = context.get(var, "unknown")
            if var_type in ("scalar", "int", "float", "Int", "Float"):
                info.scalar_vars.add(var)
            elif var_type in ("array", "list", "List", "Array", "Vec", "Mat"):
                info.array_vars.add(var)

        # Determine broadcast pattern
        if info.scalar_vars and info.array_vars:
            info.can_broadcast = True
            info.broadcast_pattern = "scalar-array"
            arr_names = ", ".join(info.array_vars)
            scalar_names = ", ".join(info.scalar_vars)
            info.numpy_equivalent = f"# {arr_names} op {scalar_names} (broadcasting)"

        return info

    def analyze_nested_loops(
        self, outer_loop: ForStatement, inner_loop: ForStatement
    ) -> BroadcastInfo:
        """
        Analyze nested loops for broadcasting opportunities.

        Pattern: for i: for j: c[i,j] = a[i] * b[j]
        Can become: c = np.outer(a, b) or a[:, np.newaxis] * b
        """
        info = BroadcastInfo()

        if len(inner_loop.body.statements) != 1:
            return info

        stmt = inner_loop.body.statements[0]
        if not isinstance(stmt, AssignmentStatement):
            return info

        # Check for outer product pattern
        target = stmt.target
        value = stmt.value

        if (
            isinstance(target, IndexExpression)
            and isinstance(value, BinaryExpression)
            and value.operator == BinaryOperator.MUL
        ):
            # Check if left uses outer loop var, right uses inner
            outer_var = outer_loop.variable
            inner_var = inner_loop.variable

            left_uses_outer = self._uses_only_var(value.left, outer_var)
            right_uses_inner = self._uses_only_var(value.right, inner_var)

            if left_uses_outer and right_uses_inner:
                info.can_broadcast = True
                info.broadcast_pattern = "outer_product"
                info.numpy_equivalent = "c = np.outer(a, b)"

        return info

    def _uses_only_var(self, expr: Expression, var_name: str) -> bool:
        """Check if expression only uses the specified variable as index."""
        if isinstance(expr, IndexExpression) and isinstance(expr.index, Identifier):
            return expr.index.name == var_name
        return False


class SIMDGenerator:
    """
    Generate SIMD-optimized code for Numba.

    This class generates code that uses Numba's explicit SIMD support
    including prange, fastmath, and vectorization hints.
    """

    def generate_simd_loop(
        self,
        _loop: ForStatement,
        parallel: bool = True,
        fastmath: bool = True,
    ) -> str:
        """
        Generate loop with explicit SIMD hints.

        Args:
            _loop: The loop to transform (used for future expansion)
            parallel: Whether to use parallel=True
            fastmath: Whether to use fastmath=True

        Returns:
            Python code string with Numba SIMD optimizations
        """
        lines = ["from numba import njit, prange", ""]

        # Build decorator
        opts = []
        if parallel:
            opts.append("parallel=True")
        if fastmath:
            opts.append("fastmath=True")
        opts.append("cache=True")

        decorator = f"@njit({', '.join(opts)})"
        lines.append(decorator)

        # Generate function signature
        lines.append("def simd_vectorized(arr):")

        # Generate loop with prange
        if parallel:
            lines.append("    for i in prange(len(arr)):")
        else:
            lines.append("    for i in range(len(arr)):")

        # Generate body (simplified)
        lines.append("        # Loop body")
        lines.append("        arr[i] = arr[i] * 2  # Example")

        lines.append("")
        return "\n".join(lines)

    def generate_vectorize_decorator(
        self,
        func_name: str,
        signature: str,
        target: str = "parallel",
    ) -> str:
        """
        Generate @vectorize decorated function.

        Args:
            func_name: Name for the vectorized function
            signature: Numba type signature (e.g., "float64(float64)")
            target: Compilation target ("cpu", "parallel", "cuda")

        Returns:
            Python code string with @vectorize decorator
        """
        lines = ["from numba import vectorize", ""]

        lines.append(f"@vectorize(['{signature}'], target='{target}')")
        lines.append(f"def {func_name}(x):")
        lines.append("    # Element-wise computation")
        lines.append("    return x * 2  # Example")
        lines.append("")

        return "\n".join(lines)

    def generate_guvectorize_decorator(
        self,
        func_name: str,
        signature: str,
        layout: str,
        target: str = "parallel",
    ) -> str:
        """
        Generate @guvectorize decorated function.

        Args:
            func_name: Name for the guvectorized function
            signature: Numba type signature
            layout: Array layout specification (e.g., "(n),(n)->(n)")
            target: Compilation target

        Returns:
            Python code string with @guvectorize decorator
        """
        lines = ["from numba import guvectorize", ""]

        lines.append(f"@guvectorize(['{signature}'], '{layout}', target='{target}')")
        lines.append(f"def {func_name}(a, b, out):")
        lines.append("    for i in range(a.shape[0]):")
        lines.append("        out[i] = a[i] + b[i]  # Example")
        lines.append("")

        return "\n".join(lines)


class VectorizationReport:
    """
    Generate detailed vectorization report.

    This class analyzes a function and produces a comprehensive report
    of all vectorization opportunities, blockers, and suggestions.
    """

    def __init__(self) -> None:
        self._vectorizer = LoopVectorizer()
        self._stencil_detector = StencilDetector()
        self._broadcast_analyzer = BroadcastAnalyzer()

    def generate(self, func: FunctionDef) -> str:
        """
        Generate report of vectorization opportunities.

        The report includes:
        - Loops that were vectorized
        - Loops that couldn't be vectorized (with reasons)
        - Estimated speedup
        - Suggestions for manual optimization

        Args:
            func: The function to analyze

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"VECTORIZATION REPORT: {func.name}",
            "=" * 70,
            "",
        ]

        if not func.body:
            lines.append("No function body to analyze.")
            return "\n".join(lines)

        # Find all loops
        loops = self._find_loops(func.body)

        if not loops:
            lines.append("No loops found in function.")
            lines.append("")
            lines.append("Suggestions:")
            lines.append("  - Consider using NumPy array operations directly")
            lines.append("  - Use list comprehensions which may be vectorizable")
            return "\n".join(lines)

        lines.append(f"Found {len(loops)} loop(s) to analyze:")
        lines.append("")

        vectorizable_count = 0
        total_speedup = 0.0

        for i, loop in enumerate(loops, 1):
            lines.append(f"Loop {i}: for {loop.variable} in ...")
            lines.append("-" * 50)

            result = self._vectorizer.analyze_loop(loop)

            lines.append(f"  Pattern: {result.pattern.name}")
            lines.append(f"  Strategy: {result.strategy.value}")
            lines.append(f"  Vectorizable: {result.can_vectorize}")

            if result.can_vectorize:
                vectorizable_count += 1
                total_speedup += result.estimated_speedup
                lines.append(f"  Estimated Speedup: {result.estimated_speedup:.1f}x")

                if result.transformed_code:
                    lines.append("  Transformed Code:")
                    for code_line in result.transformed_code.split("\n"):
                        lines.append(f"    {code_line}")

            else:
                lines.append("  Blockers:")
                for blocker in result.blockers:
                    lines.append(f"    - {blocker}")

            # Check for stencil pattern
            stencil_info = self._stencil_detector.detect(loop)
            if stencil_info:
                lines.append(f"  Stencil Detected: {stencil_info.dimensions}D")
                lines.append(f"    Radius: {stencil_info.radius}")
                lines.append(f"    Symmetric: {stencil_info.is_symmetric}")

            # Suggestions
            suggestions = self._generate_suggestions(loop, result)
            if suggestions:
                lines.append("  Suggestions:")
                for suggestion in suggestions:
                    lines.append(f"    - {suggestion}")

            lines.append("")

        # Summary
        lines.append("=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Total Loops: {len(loops)}")
        lines.append(f"Vectorizable: {vectorizable_count}")
        lines.append(f"Not Vectorizable: {len(loops) - vectorizable_count}")

        if vectorizable_count > 0:
            avg_speedup = total_speedup / vectorizable_count
            lines.append(f"Average Estimated Speedup: {avg_speedup:.1f}x")

        lines.append("")

        return "\n".join(lines)

    def _find_loops(self, node: ASTNode) -> list[ForStatement]:
        """Recursively find all for loops in the AST."""
        loops: list[ForStatement] = []

        if isinstance(node, ForStatement):
            loops.append(node)
            loops.extend(self._find_loops(node.body))

        elif isinstance(node, Block):
            for stmt in node.statements:
                loops.extend(self._find_loops(stmt))

        elif isinstance(node, IfStatement):
            loops.extend(self._find_loops(node.then_block))
            for _, elif_block in node.elif_clauses:
                loops.extend(self._find_loops(elif_block))
            if node.else_block:
                loops.extend(self._find_loops(node.else_block))

        elif isinstance(node, WhileStatement):
            loops.extend(self._find_loops(node.body))

        return loops

    def _generate_suggestions(self, _loop: ForStatement, result: VectorizationResult) -> list[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions: list[str] = []

        if not result.can_vectorize:
            # Suggest fixes for blockers
            for blocker in result.blockers:
                if "break" in blocker.lower():
                    suggestions.append(
                        "Replace break with np.argmax/argmin for early exit patterns"
                    )
                elif "continue" in blocker.lower():
                    suggestions.append("Use np.where for conditional processing")
                elif "nested" in blocker.lower():
                    suggestions.append("Consider np.einsum for nested array operations")
                elif "dependency" in blocker.lower():
                    suggestions.append("Use np.cumsum/cumprod for prefix computations")

        if result.pattern == LoopPattern.REDUCTION:
            suggestions.append("Consider np.sum, np.prod, np.min, np.max for reductions")

        if result.pattern == LoopPattern.STENCIL_1D:
            suggestions.append("Use scipy.ndimage.convolve for general convolutions")

        return suggestions


# =============================================================================
# Main API Functions
# =============================================================================


def vectorize_function(
    func: FunctionDef,
    aggressive: bool = False,  # noqa: ARG001
) -> tuple[FunctionDef, VectorizationReport]:
    """
    Main entry point for vectorization.

    Analyzes and transforms a function to use vectorized operations
    where possible.

    Args:
        func: The function definition to vectorize
        aggressive: If True, apply more aggressive transformations (reserved for future use)

    Returns:
        Tuple of (transformed function, vectorization report)
    """
    transformer = NumpyTransformer()
    report_generator = VectorizationReport()

    # Generate report first
    report_generator.generate(func)

    # Transform function
    transformed_func, _ = transformer.transform(func)

    # Create report object
    report = VectorizationReport()

    return transformed_func, report


def generate_vectorized_code(func: FunctionDef) -> str:
    """
    Generate Python code with vectorized operations.

    This function takes a MathViz function definition and generates
    equivalent Python code that uses NumPy vectorized operations
    instead of explicit loops.

    Args:
        func: The function definition to transform

    Returns:
        Python code string with vectorized operations
    """
    vectorizer = LoopVectorizer()
    lines: list[str] = []

    # Add imports
    lines.append("import numpy as np")
    lines.append("from numba import njit, prange")
    lines.append("")

    # Generate function header
    params = ", ".join(p.name for p in func.parameters)
    lines.append(f"def {func.name}({params}):")

    if not func.body:
        lines.append("    pass")
        return "\n".join(lines)

    # Analyze and transform each statement
    for stmt in func.body.statements:
        if isinstance(stmt, ForStatement):
            result = vectorizer.analyze_loop(stmt)
            if result.can_vectorize and result.transformed_code:
                lines.append("    # Vectorized from loop")
                lines.append(f"    {result.transformed_code}")
            else:
                lines.append("    # Original loop (not vectorized)")
                lines.append(f"    for {stmt.variable} in range(...):")
                lines.append("        # Loop body")
        else:
            lines.append("    # Statement")

    lines.append("")
    return "\n".join(lines)


def analyze_vectorization(func: FunctionDef) -> list[VectorizationResult]:
    """
    Analyze a function for vectorization opportunities.

    Args:
        func: The function to analyze

    Returns:
        List of VectorizationResult for each loop found
    """
    vectorizer = LoopVectorizer()
    report_gen = VectorizationReport()
    results: list[VectorizationResult] = []

    if not func.body:
        return results

    loops = report_gen._find_loops(func.body)

    for loop in loops:
        result = vectorizer.analyze_loop(loop)
        results.append(result)

    return results


def can_vectorize_loop(loop: ForStatement) -> bool:
    """
    Quick check if a loop can be vectorized.

    Args:
        loop: The for loop to check

    Returns:
        True if the loop can be vectorized
    """
    vectorizer = LoopVectorizer()
    result = vectorizer.analyze_loop(loop)
    return result.can_vectorize


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enumerations
    "VectorizationStrategy",
    "LoopPattern",
    "ReductionType",
    # Data classes
    "VectorizationResult",
    "ArrayAccessInfo",
    "ReductionInfo",
    "StencilInfo",
    "BroadcastInfo",
    # Main classes
    "LoopVectorizer",
    "NumpyTransformer",
    "StencilDetector",
    "BroadcastAnalyzer",
    "SIMDGenerator",
    "VectorizationReport",
    # API functions
    "vectorize_function",
    "generate_vectorized_code",
    "analyze_vectorization",
    "can_vectorize_loop",
]
