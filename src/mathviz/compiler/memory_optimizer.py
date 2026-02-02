"""
Memory Optimization Pass for MathViz Compiler.

This module optimizes memory allocation, access patterns, and reduces unnecessary
allocations. It performs several analyses and transformations:

1. Allocation Analysis - Find and track array/buffer allocations
2. Buffer Reuse - Reuse buffers with non-overlapping lifetimes
3. In-Place Operation Detection - Convert operations to in-place where safe
4. Cache Optimization - Optimize for CPU cache efficiency
5. Memory Layout Optimization - Suggest optimal memory layouts
6. Temporary Elimination - Eliminate unnecessary temporary arrays
7. Memory Pool Generation - Generate memory pools for repeated allocations

The optimizer is conservative: transformations are only applied when they can be
proven safe. Correctness is never sacrificed for performance.

References:
- Allen & Kennedy, "Optimizing Compilers for Modern Architectures"
- Intel Optimization Manual
- NumPy Memory Layout Documentation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from mathviz.compiler.ast_nodes import (
    # AST base classes
    ASTNode,
    BaseASTVisitor,
    Expression,
    Statement,
    Block,
    # Expressions
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    MemberAccess,
    IndexExpression,
    ConditionalExpression,
    RangeExpression,
    BinaryOperator,
    UnaryOperator,
    # Statements
    LetStatement,
    AssignmentStatement,
    CompoundAssignment,
    FunctionDef,
    ForStatement,
    WhileStatement,
    IfStatement,
    ReturnStatement,
    ExpressionStatement,
    Parameter,
)
from mathviz.utils.errors import SourceLocation


# =============================================================================
# Data Structures
# =============================================================================


class AccessPattern(Enum):
    """Memory access pattern classification."""

    SEQUENTIAL = auto()    # arr[i], arr[i+1], ...
    STRIDED = auto()       # arr[i*2], arr[i*stride], ...
    RANDOM = auto()        # arr[indices[i]], unpredictable
    COLUMN_MAJOR = auto()  # arr[i][j] with j as outer loop
    ROW_MAJOR = auto()     # arr[i][j] with i as outer loop
    UNKNOWN = auto()       # Cannot determine


class MemoryOrder(Enum):
    """Memory layout order."""

    C_ORDER = "C"          # Row-major (C-style)
    FORTRAN_ORDER = "F"    # Column-major (Fortran-style)
    UNKNOWN = "?"          # Cannot determine


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """
    Source location information for allocations.

    Note: This shadows the import but provides a simplified version
    for memory optimization context.
    """

    line: int
    column: int = 0
    filename: Optional[str] = None

    def __str__(self) -> str:
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"


@dataclass(slots=True)
class AllocationInfo:
    """
    Information about an array/buffer allocation.

    Tracks allocation details including size, type, location, and lifetime
    for buffer reuse analysis.

    Attributes:
        variable: Name of the allocated variable
        size_expr: Expression representing the size
        element_type: Type of array elements (e.g., "Float", "Int")
        location: Source location of allocation
        is_temporary: True if allocation is a temporary (intermediate result)
        lifetime: Tuple of (start_line, end_line) for the variable's lifetime
        can_reuse: True if this buffer can be reused by another allocation
        allocation_func: The function used for allocation (zeros, ones, empty)
        shape_dims: Number of dimensions
        estimated_size: Estimated size in elements (if computable)
    """

    variable: str
    size_expr: Optional[Expression]
    element_type: str
    location: Optional[SourceLocation]
    is_temporary: bool
    lifetime: tuple[int, int]
    can_reuse: bool
    allocation_func: str = "zeros"
    shape_dims: int = 1
    estimated_size: Optional[int] = None

    def __str__(self) -> str:
        temp_str = " (temporary)" if self.is_temporary else ""
        reuse_str = " [reusable]" if self.can_reuse else ""
        return (
            f"{self.variable}: {self.allocation_func}[{self.element_type}] "
            f"lifetime={self.lifetime}{temp_str}{reuse_str}"
        )


@dataclass(slots=True)
class ArrayAccess:
    """
    Represents an array access for pattern analysis.

    Attributes:
        array_var: Name of the array being accessed
        indices: List of index expressions (for multi-dimensional)
        is_write: True if this is a write access
        loop_var: Loop variable if access is inside a loop
        index_depends_on_loop: True if index depends on loop variable
        stride: Detected stride relative to loop variable
        location: Source location
    """

    array_var: str
    indices: list[Expression]
    is_write: bool
    loop_var: Optional[str] = None
    index_depends_on_loop: bool = False
    stride: int = 1
    location: Optional[SourceLocation] = None


@dataclass(slots=True)
class CacheAnalysis:
    """
    Cache behavior analysis results.

    Attributes:
        access_pattern: Detected memory access pattern
        estimated_cache_misses: Estimated cache misses (relative scale)
        suggestions: List of optimization suggestions
        loop_var: The analyzed loop variable
        arrays_accessed: Arrays accessed in the loop
        spatial_locality: Score from 0-1 for spatial locality
        temporal_locality: Score from 0-1 for temporal locality
    """

    access_pattern: str
    estimated_cache_misses: int
    suggestions: list[str]
    loop_var: str = ""
    arrays_accessed: list[str] = field(default_factory=list)
    spatial_locality: float = 1.0
    temporal_locality: float = 1.0

    def __str__(self) -> str:
        lines = [
            f"Cache Analysis: {self.access_pattern}",
            f"  Estimated cache misses: {self.estimated_cache_misses}",
            f"  Spatial locality: {self.spatial_locality:.2f}",
            f"  Temporal locality: {self.temporal_locality:.2f}",
        ]
        if self.suggestions:
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")
        return "\n".join(lines)


@dataclass(slots=True)
class LoopInterchange:
    """
    Suggested loop interchange optimization.

    Attributes:
        outer_var: Current outer loop variable
        inner_var: Current inner loop variable
        reason: Explanation for the interchange
        expected_improvement: Expected improvement factor
    """

    outer_var: str
    inner_var: str
    reason: str
    expected_improvement: float = 1.0

    def __str__(self) -> str:
        return (
            f"Loop interchange: swap {self.outer_var} (outer) with {self.inner_var} (inner) "
            f"[{self.expected_improvement:.1f}x improvement] - {self.reason}"
        )


@dataclass(slots=True)
class BlockingInfo:
    """
    Loop blocking/tiling information.

    Attributes:
        loop_var: Variable to block
        block_size: Suggested block size
        reason: Explanation for blocking
    """

    loop_var: str
    block_size: int
    reason: str

    def __str__(self) -> str:
        return f"Block loop {self.loop_var} with size {self.block_size}: {self.reason}"


@dataclass(slots=True)
class InPlaceCandidate:
    """
    A candidate for in-place operation transformation.

    Attributes:
        variable: Target variable
        operation: The operation being performed
        source_vars: Variables used as sources
        location: Source location
        transform_type: Type of transformation ("compound", "out_param", "inplace_method")
        original_code: Original code string (for reporting)
        suggested_code: Suggested transformation
    """

    variable: str
    operation: str
    source_vars: list[str]
    location: Optional[SourceLocation]
    transform_type: str
    original_code: str = ""
    suggested_code: str = ""


# =============================================================================
# Graph Data Structure for Buffer Reuse
# =============================================================================


@dataclass(slots=True)
class Graph:
    """
    Simple graph data structure for interference graph.

    Nodes are identified by string names (variable names).
    Edges represent interference (cannot share memory).
    """

    nodes: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, name: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(name)
        if name not in self.edges:
            self.edges[name] = set()

    def add_edge(self, node1: str, node2: str) -> None:
        """Add an undirected edge between two nodes."""
        self.add_node(node1)
        self.add_node(node2)
        self.edges[node1].add(node2)
        self.edges[node2].add(node1)

    def has_edge(self, node1: str, node2: str) -> bool:
        """Check if an edge exists between two nodes."""
        return node1 in self.edges and node2 in self.edges[node1]

    def neighbors(self, node: str) -> set[str]:
        """Get all neighbors of a node."""
        return self.edges.get(node, set())

    def degree(self, node: str) -> int:
        """Get the degree (number of edges) of a node."""
        return len(self.edges.get(node, set()))


# =============================================================================
# Allocation Analysis
# =============================================================================


class AllocationAnalyzer(BaseASTVisitor):
    """
    Analyzes functions to find all array/buffer allocations.

    Detects allocations from:
    - np.zeros, np.ones, np.empty
    - np.full, np.zeros_like, np.ones_like, np.empty_like
    - Array constructors

    Also computes variable lifetimes for buffer reuse analysis.
    """

    # Known allocation functions
    ALLOCATION_FUNCTIONS: frozenset[str] = frozenset({
        "zeros", "ones", "empty", "full",
        "zeros_like", "ones_like", "empty_like",
        "np.zeros", "np.ones", "np.empty", "np.full",
        "np.zeros_like", "np.ones_like", "np.empty_like",
        "arange", "linspace", "np.arange", "np.linspace",
        "eye", "identity", "np.eye", "np.identity",
    })

    def __init__(self) -> None:
        """Initialize the allocation analyzer."""
        self._allocations: list[AllocationInfo] = []
        self._current_line: int = 0
        self._variable_definitions: dict[str, int] = {}  # var -> definition line
        self._variable_uses: dict[str, list[int]] = {}   # var -> list of use lines
        self._local_vars: set[str] = set()
        self._parameters: set[str] = set()

    def analyze(self, func: FunctionDef) -> list[AllocationInfo]:
        """
        Find all array/buffer allocations in a function.

        Args:
            func: The function to analyze

        Returns:
            List of AllocationInfo for each allocation found
        """
        self._allocations = []
        self._current_line = func.location.line if func.location else 1
        self._variable_definitions = {}
        self._variable_uses = {}
        self._local_vars = set()
        self._parameters = {p.name for p in func.parameters}

        if func.body:
            self._analyze_block(func.body)
            self._compute_lifetimes(self._allocations, func)

        return self._allocations

    def _analyze_block(self, block: Block) -> None:
        """Analyze a block of statements."""
        for stmt in block.statements:
            if stmt.location:
                self._current_line = stmt.location.line
            self._analyze_statement(stmt)

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a single statement for allocations."""
        if isinstance(stmt, LetStatement):
            self._local_vars.add(stmt.name)
            self._variable_definitions[stmt.name] = self._current_line
            if stmt.value:
                self._check_allocation(stmt.name, stmt.value, stmt.location)
                self._collect_variable_uses(stmt.value)

        elif isinstance(stmt, AssignmentStatement):
            if isinstance(stmt.target, Identifier):
                self._variable_definitions[stmt.target.name] = self._current_line
                self._check_allocation(stmt.target.name, stmt.value, stmt.location)
            self._collect_variable_uses(stmt.value)

        elif isinstance(stmt, ForStatement):
            self._local_vars.add(stmt.variable)
            self._collect_variable_uses(stmt.iterable)
            self._analyze_block(stmt.body)

        elif isinstance(stmt, WhileStatement):
            self._collect_variable_uses(stmt.condition)
            self._analyze_block(stmt.body)

        elif isinstance(stmt, IfStatement):
            self._collect_variable_uses(stmt.condition)
            self._analyze_block(stmt.then_block)
            for cond, block in stmt.elif_clauses:
                self._collect_variable_uses(cond)
                self._analyze_block(block)
            if stmt.else_block:
                self._analyze_block(stmt.else_block)

        elif isinstance(stmt, ReturnStatement):
            if stmt.value:
                self._collect_variable_uses(stmt.value)

        elif isinstance(stmt, ExpressionStatement):
            self._collect_variable_uses(stmt.expression)

    def _check_allocation(
        self,
        var_name: str,
        expr: Expression,
        location: Optional[SourceLocation],
    ) -> None:
        """Check if an expression is an allocation and record it."""
        if not isinstance(expr, CallExpression):
            return

        func_name = self._get_function_name(expr.callee)
        if func_name not in self.ALLOCATION_FUNCTIONS:
            return

        # Extract allocation information
        size_expr = expr.arguments[0] if expr.arguments else None
        element_type = self._infer_element_type(func_name, expr)
        is_temporary = var_name.startswith("_") or var_name.startswith("tmp")

        # Estimate size if possible
        estimated_size = self._estimate_size(size_expr) if size_expr else None

        # Determine number of dimensions
        shape_dims = self._get_shape_dims(size_expr)

        # Create allocation info (lifetime will be computed later)
        alloc_info = AllocationInfo(
            variable=var_name,
            size_expr=size_expr,
            element_type=element_type,
            location=location,
            is_temporary=is_temporary,
            lifetime=(self._current_line, self._current_line),  # Placeholder
            can_reuse=True,  # Will be refined during lifetime analysis
            allocation_func=func_name.split(".")[-1],  # Remove np. prefix
            shape_dims=shape_dims,
            estimated_size=estimated_size,
        )
        self._allocations.append(alloc_info)

    def _get_function_name(self, callee: Expression) -> str:
        """Extract function name from callee expression."""
        if isinstance(callee, Identifier):
            return callee.name
        elif isinstance(callee, MemberAccess):
            base = self._get_base_name(callee.object)
            return f"{base}.{callee.member}" if base else callee.member
        return ""

    def _get_base_name(self, expr: Expression) -> Optional[str]:
        """Get base identifier name from expression."""
        if isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, MemberAccess):
            return self._get_base_name(expr.object)
        return None

    def _infer_element_type(self, func_name: str, call: CallExpression) -> str:
        """Infer element type from allocation call."""
        # Check for dtype argument
        # For now, default to Float64
        base_name = func_name.split(".")[-1]
        if base_name in ("zeros", "ones", "empty", "full"):
            return "Float"  # Default numpy dtype
        elif "int" in func_name.lower():
            return "Int"
        return "Float"

    def _estimate_size(self, size_expr: Expression) -> Optional[int]:
        """Try to estimate the allocation size."""
        if isinstance(size_expr, IntegerLiteral):
            return size_expr.value
        elif isinstance(size_expr, BinaryExpression):
            left = self._estimate_size(size_expr.left)
            right = self._estimate_size(size_expr.right)
            if left is not None and right is not None:
                if size_expr.operator == BinaryOperator.MUL:
                    return left * right
                elif size_expr.operator == BinaryOperator.ADD:
                    return left + right
        return None

    def _get_shape_dims(self, size_expr: Optional[Expression]) -> int:
        """Determine number of dimensions from size expression."""
        if size_expr is None:
            return 1
        # Tuple indicates multi-dimensional
        # For now, assume 1D unless we can detect otherwise
        return 1

    def _collect_variable_uses(self, expr: Expression) -> None:
        """Collect all variable uses in an expression."""
        if isinstance(expr, Identifier):
            name = expr.name
            if name not in self._variable_uses:
                self._variable_uses[name] = []
            self._variable_uses[name].append(self._current_line)

        elif isinstance(expr, BinaryExpression):
            self._collect_variable_uses(expr.left)
            self._collect_variable_uses(expr.right)

        elif isinstance(expr, UnaryExpression):
            self._collect_variable_uses(expr.operand)

        elif isinstance(expr, CallExpression):
            self._collect_variable_uses(expr.callee)
            for arg in expr.arguments:
                self._collect_variable_uses(arg)

        elif isinstance(expr, IndexExpression):
            self._collect_variable_uses(expr.object)
            self._collect_variable_uses(expr.index)

        elif isinstance(expr, MemberAccess):
            self._collect_variable_uses(expr.object)

        elif isinstance(expr, ConditionalExpression):
            self._collect_variable_uses(expr.condition)
            self._collect_variable_uses(expr.then_expr)
            self._collect_variable_uses(expr.else_expr)

    def _compute_lifetimes(
        self,
        allocations: list[AllocationInfo],
        func: FunctionDef,
    ) -> None:
        """Compute variable lifetimes for reuse analysis."""
        for alloc in allocations:
            var_name = alloc.variable

            # Start of lifetime is the definition
            start_line = self._variable_definitions.get(var_name, alloc.lifetime[0])

            # End of lifetime is the last use
            uses = self._variable_uses.get(var_name, [])
            end_line = max(uses) if uses else start_line

            # Update the allocation info
            alloc.lifetime = (start_line, end_line)

            # Determine if buffer can be reused
            # Can reuse if:
            # 1. It's a temporary
            # 2. It's not returned from the function
            # 3. It's not passed to external functions after last use
            alloc.can_reuse = alloc.is_temporary

    def _find_zeros_ones_empty(self, func: FunctionDef) -> list[AllocationInfo]:
        """
        Find np.zeros, np.ones, np.empty calls specifically.

        This is a convenience method that filters results.
        """
        all_allocs = self.analyze(func)
        target_funcs = {"zeros", "ones", "empty"}
        return [
            a for a in all_allocs
            if a.allocation_func in target_funcs
        ]


# =============================================================================
# Buffer Reuse Optimizer
# =============================================================================


class BufferReuseOptimizer:
    """
    Reuses buffers with non-overlapping lifetimes.

    Uses graph coloring to find an optimal assignment of buffers to
    memory slots, minimizing total memory usage while ensuring correctness.
    """

    def __init__(self) -> None:
        """Initialize the buffer reuse optimizer."""
        self._allocation_analyzer = AllocationAnalyzer()

    def optimize(self, func: FunctionDef) -> tuple[FunctionDef, list[str]]:
        """
        Optimize buffer allocations by reusing buffers.

        Args:
            func: The function to optimize

        Returns:
            Tuple of (potentially modified function, list of optimization messages)
        """
        # Analyze allocations
        allocations = self._allocation_analyzer.analyze(func)

        if len(allocations) < 2:
            return func, []

        # Filter to only reusable allocations of the same type
        reusable = [a for a in allocations if a.can_reuse]

        if len(reusable) < 2:
            return func, []

        # Group allocations by compatible types
        type_groups: dict[str, list[AllocationInfo]] = {}
        for alloc in reusable:
            key = f"{alloc.element_type}_{alloc.shape_dims}"
            if key not in type_groups:
                type_groups[key] = []
            type_groups[key].append(alloc)

        messages: list[str] = []

        # Process each type group
        for type_key, group_allocs in type_groups.items():
            if len(group_allocs) < 2:
                continue

            # Build interference graph
            graph = self._build_interference_graph(group_allocs)

            # Color the graph
            coloring = self._color_graph(graph)

            # Count unique colors used
            num_colors = len(set(coloring.values()))

            if num_colors < len(group_allocs):
                messages.append(
                    f"Can reuse buffers: {len(group_allocs)} allocations "
                    f"of type {type_key} can share {num_colors} buffer(s)"
                )

                # Report which buffers can share memory
                color_groups: dict[int, list[str]] = {}
                for var, color in coloring.items():
                    if color not in color_groups:
                        color_groups[color] = []
                    color_groups[color].append(var)

                for color, vars in color_groups.items():
                    if len(vars) > 1:
                        messages.append(
                            f"  Buffer slot {color}: {', '.join(vars)} can share memory"
                        )

        # Note: Actual AST transformation would require creating new nodes
        # For now, we return the original function with optimization suggestions
        return func, messages

    def _build_interference_graph(
        self,
        allocations: list[AllocationInfo],
    ) -> Graph:
        """
        Build graph of buffers that cannot share memory.

        Two buffers interfere if their lifetimes overlap.
        """
        graph = Graph()

        for alloc in allocations:
            graph.add_node(alloc.variable)

        # Add edges for interfering allocations
        for i, alloc1 in enumerate(allocations):
            for alloc2 in allocations[i + 1:]:
                if self._lifetimes_overlap(alloc1.lifetime, alloc2.lifetime):
                    graph.add_edge(alloc1.variable, alloc2.variable)

        return graph

    def _lifetimes_overlap(
        self,
        lifetime1: tuple[int, int],
        lifetime2: tuple[int, int],
    ) -> bool:
        """Check if two lifetimes overlap."""
        start1, end1 = lifetime1
        start2, end2 = lifetime2
        return not (end1 < start2 or end2 < start1)

    def _color_graph(self, graph: Graph) -> dict[str, int]:
        """
        Color graph to assign buffer slots using greedy coloring.

        Uses the Welsh-Powell algorithm for efficient greedy coloring.
        """
        if not graph.nodes:
            return {}

        # Sort nodes by degree (descending)
        sorted_nodes = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)

        coloring: dict[str, int] = {}
        for node in sorted_nodes:
            # Find colors used by neighbors
            neighbor_colors = {
                coloring[n] for n in graph.neighbors(node)
                if n in coloring
            }

            # Find smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[node] = color

        return coloring

    def _generate_buffer_pool(self, allocations: list[AllocationInfo]) -> str:
        """
        Generate buffer pool code for a set of allocations.

        Creates a class that pre-allocates buffers for reuse.
        """
        if not allocations:
            return ""

        # Find max size needed for each buffer slot
        coloring = self._color_graph(self._build_interference_graph(allocations))

        # Group by color and find max size
        slot_sizes: dict[int, int] = {}
        slot_types: dict[int, str] = {}

        for alloc in allocations:
            color = coloring.get(alloc.variable, 0)
            size = alloc.estimated_size or 1000  # Default size

            if color not in slot_sizes or size > slot_sizes[color]:
                slot_sizes[color] = size
                slot_types[color] = alloc.element_type

        # Generate code
        lines = [
            "class _MemPool:",
            "    '''Pre-allocated buffer pool for memory optimization.'''",
            "",
            "    def __init__(self, n: int) -> None:",
        ]

        for slot, size in sorted(slot_sizes.items()):
            dtype = "np.float64" if slot_types[slot] == "Float" else "np.int64"
            lines.append(f"        self.buf{slot} = np.empty(n, dtype={dtype})")

        lines.append("")
        lines.append("    def reset(self) -> None:")
        lines.append("        '''Reset all buffers (optional, for debugging).'''")
        lines.append("        pass")

        return "\n".join(lines)


# =============================================================================
# In-Place Operation Detection
# =============================================================================


class InPlaceOptimizer(BaseASTVisitor):
    """
    Detects operations that can be done in-place.

    Transformations:
    - a = a + b  ->  a += b  (or np.add(a, b, out=a))
    - result = np.sqrt(arr)  ->  np.sqrt(arr, out=result) if result pre-allocated
    """

    # Operations that support in-place versions
    INPLACE_OPS: dict[BinaryOperator, str] = {
        BinaryOperator.ADD: "+=",
        BinaryOperator.SUB: "-=",
        BinaryOperator.MUL: "*=",
        BinaryOperator.DIV: "/=",
    }

    # NumPy functions that support out= parameter
    NUMPY_OUT_FUNCS: frozenset[str] = frozenset({
        "sqrt", "sin", "cos", "tan", "exp", "log", "log10", "log2",
        "abs", "floor", "ceil", "round",
        "add", "subtract", "multiply", "divide", "power",
        "negative", "positive", "square", "reciprocal",
        "np.sqrt", "np.sin", "np.cos", "np.tan", "np.exp",
        "np.log", "np.log10", "np.log2", "np.abs",
        "np.floor", "np.ceil", "np.round",
        "np.add", "np.subtract", "np.multiply", "np.divide", "np.power",
    })

    def __init__(self) -> None:
        """Initialize the in-place optimizer."""
        self._candidates: list[InPlaceCandidate] = []
        self._pre_allocated: set[str] = set()  # Variables that are pre-allocated
        self._variable_last_use: dict[str, int] = {}
        self._current_line: int = 0

    def optimize(self, func: FunctionDef) -> tuple[FunctionDef, list[InPlaceCandidate]]:
        """
        Convert operations to in-place where safe.

        Args:
            func: Function to optimize

        Returns:
            Tuple of (function, list of in-place candidates)
        """
        self._candidates = []
        self._pre_allocated = set()
        self._variable_last_use = {}
        self._current_line = 0

        # First pass: find pre-allocated buffers and last uses
        if func.body:
            self._analyze_allocations_and_uses(func.body)

        # Second pass: find in-place candidates
        if func.body:
            self._find_inplace_candidates(func.body)

        return func, self._candidates

    def _analyze_allocations_and_uses(self, block: Block) -> None:
        """Find pre-allocated buffers and track last uses."""
        for stmt in block.statements:
            if stmt.location:
                self._current_line = stmt.location.line

            if isinstance(stmt, LetStatement):
                if stmt.value and self._is_allocation(stmt.value):
                    self._pre_allocated.add(stmt.name)
                if stmt.value:
                    self._track_uses(stmt.value)

            elif isinstance(stmt, AssignmentStatement):
                self._track_uses(stmt.value)
                if isinstance(stmt.target, Identifier):
                    self._track_uses(stmt.target)

            elif isinstance(stmt, ForStatement):
                self._track_uses(stmt.iterable)
                self._analyze_allocations_and_uses(stmt.body)

            elif isinstance(stmt, IfStatement):
                self._track_uses(stmt.condition)
                self._analyze_allocations_and_uses(stmt.then_block)
                for cond, blk in stmt.elif_clauses:
                    self._track_uses(cond)
                    self._analyze_allocations_and_uses(blk)
                if stmt.else_block:
                    self._analyze_allocations_and_uses(stmt.else_block)

    def _is_allocation(self, expr: Expression) -> bool:
        """Check if expression is an allocation."""
        if not isinstance(expr, CallExpression):
            return False
        func_name = self._get_func_name(expr.callee)
        return func_name in AllocationAnalyzer.ALLOCATION_FUNCTIONS

    def _get_func_name(self, callee: Expression) -> str:
        """Get function name from callee."""
        if isinstance(callee, Identifier):
            return callee.name
        elif isinstance(callee, MemberAccess):
            base = self._get_base(callee.object)
            return f"{base}.{callee.member}" if base else callee.member
        return ""

    def _get_base(self, expr: Expression) -> Optional[str]:
        """Get base identifier."""
        if isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, MemberAccess):
            return self._get_base(expr.object)
        return None

    def _track_uses(self, expr: Expression) -> None:
        """Track variable uses."""
        if isinstance(expr, Identifier):
            self._variable_last_use[expr.name] = self._current_line
        elif isinstance(expr, BinaryExpression):
            self._track_uses(expr.left)
            self._track_uses(expr.right)
        elif isinstance(expr, UnaryExpression):
            self._track_uses(expr.operand)
        elif isinstance(expr, CallExpression):
            for arg in expr.arguments:
                self._track_uses(arg)
        elif isinstance(expr, IndexExpression):
            self._track_uses(expr.object)
            self._track_uses(expr.index)

    def _find_inplace_candidates(self, block: Block) -> None:
        """Find candidates for in-place operations."""
        for stmt in block.statements:
            if stmt.location:
                self._current_line = stmt.location.line

            if isinstance(stmt, AssignmentStatement):
                self._check_inplace_assignment(stmt)

            elif isinstance(stmt, ForStatement):
                self._find_inplace_candidates(stmt.body)

            elif isinstance(stmt, IfStatement):
                self._find_inplace_candidates(stmt.then_block)
                for _, blk in stmt.elif_clauses:
                    self._find_inplace_candidates(blk)
                if stmt.else_block:
                    self._find_inplace_candidates(stmt.else_block)

    def _check_inplace_assignment(self, stmt: AssignmentStatement) -> None:
        """Check if assignment can be converted to in-place."""
        if not isinstance(stmt.target, Identifier):
            return

        target_name = stmt.target.name
        value = stmt.value

        # Pattern 1: a = a + b  ->  a += b
        if isinstance(value, BinaryExpression):
            if value.operator in self.INPLACE_OPS:
                if self._can_use_compound(target_name, value):
                    op_str = self.INPLACE_OPS[value.operator]
                    self._candidates.append(InPlaceCandidate(
                        variable=target_name,
                        operation=str(value.operator),
                        source_vars=self._collect_vars(value),
                        location=stmt.location,
                        transform_type="compound",
                        original_code=f"{target_name} = {target_name} {op_str[0]} ...",
                        suggested_code=f"{target_name} {op_str} ...",
                    ))

        # Pattern 2: result = np.func(arr)  ->  np.func(arr, out=result)
        elif isinstance(value, CallExpression):
            func_name = self._get_func_name(value.callee)
            if func_name in self.NUMPY_OUT_FUNCS:
                if target_name in self._pre_allocated:
                    self._candidates.append(InPlaceCandidate(
                        variable=target_name,
                        operation=func_name,
                        source_vars=self._collect_vars(value),
                        location=stmt.location,
                        transform_type="out_param",
                        original_code=f"{target_name} = {func_name}(...)",
                        suggested_code=f"{func_name}(..., out={target_name})",
                    ))

    def _can_use_compound(self, target: str, expr: BinaryExpression) -> bool:
        """Check if expression can be converted to compound assignment."""
        # Check if target appears on the left side of operation
        if isinstance(expr.left, Identifier) and expr.left.name == target:
            return True
        # For commutative operations, also check right side
        if expr.operator in (BinaryOperator.ADD, BinaryOperator.MUL):
            if isinstance(expr.right, Identifier) and expr.right.name == target:
                return True
        return False

    def _collect_vars(self, expr: Expression) -> list[str]:
        """Collect all variable names in expression."""
        vars: list[str] = []
        if isinstance(expr, Identifier):
            vars.append(expr.name)
        elif isinstance(expr, BinaryExpression):
            vars.extend(self._collect_vars(expr.left))
            vars.extend(self._collect_vars(expr.right))
        elif isinstance(expr, UnaryExpression):
            vars.extend(self._collect_vars(expr.operand))
        elif isinstance(expr, CallExpression):
            for arg in expr.arguments:
                vars.extend(self._collect_vars(arg))
        elif isinstance(expr, IndexExpression):
            vars.extend(self._collect_vars(expr.object))
        return vars

    def _can_use_inplace(self, assignment: AssignmentStatement) -> bool:
        """Check if assignment can be done in-place."""
        if not isinstance(assignment.target, Identifier):
            return False

        target = assignment.target.name
        value = assignment.value

        # Check pattern: a = a op b
        if isinstance(value, BinaryExpression):
            if value.operator in self.INPLACE_OPS:
                return self._can_use_compound(target, value)

        return False

    def _is_last_use(self, var: str, location: int, func: FunctionDef) -> bool:
        """Check if this is the last use of variable."""
        last_use = self._variable_last_use.get(var, 0)
        return location >= last_use


# =============================================================================
# Cache Optimization
# =============================================================================


class CacheOptimizer(BaseASTVisitor):
    """
    Optimizes for CPU cache efficiency.

    Analyzes memory access patterns and suggests:
    - Loop interchange for better locality
    - Loop blocking/tiling
    - Array layout changes
    """

    # Typical L1 cache size in elements (32KB / 8 bytes)
    L1_CACHE_SIZE: int = 4096
    # Cache line size in elements (64 bytes / 8 bytes)
    CACHE_LINE_SIZE: int = 8

    def __init__(self) -> None:
        """Initialize the cache optimizer."""
        self._array_accesses: list[ArrayAccess] = []
        self._current_loop_var: Optional[str] = None
        self._nested_loops: list[str] = []

    def analyze(self, func: FunctionDef) -> CacheAnalysis:
        """
        Analyze cache behavior of function.

        Args:
            func: Function to analyze

        Returns:
            CacheAnalysis with detected patterns and suggestions
        """
        self._array_accesses = []
        self._current_loop_var = None
        self._nested_loops = []

        if func.body:
            self._analyze_block(func.body)

        return self._generate_analysis()

    def _analyze_block(self, block: Block) -> None:
        """Analyze a block for memory access patterns."""
        for stmt in block.statements:
            self._analyze_statement(stmt)

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a statement for memory accesses."""
        if isinstance(stmt, ForStatement):
            # Track nested loop structure
            outer_loops = self._nested_loops.copy()
            outer_var = self._current_loop_var

            self._nested_loops.append(stmt.variable)
            self._current_loop_var = stmt.variable

            # Analyze loop body
            self._analyze_block(stmt.body)

            self._nested_loops = outer_loops
            self._current_loop_var = outer_var

        elif isinstance(stmt, AssignmentStatement):
            self._analyze_expression(stmt.value, is_write=False)
            self._analyze_target(stmt.target)

        elif isinstance(stmt, LetStatement):
            if stmt.value:
                self._analyze_expression(stmt.value, is_write=False)

        elif isinstance(stmt, IfStatement):
            self._analyze_expression(stmt.condition, is_write=False)
            self._analyze_block(stmt.then_block)
            for cond, blk in stmt.elif_clauses:
                self._analyze_expression(cond, is_write=False)
                self._analyze_block(blk)
            if stmt.else_block:
                self._analyze_block(stmt.else_block)

    def _analyze_target(self, target: Expression) -> None:
        """Analyze assignment target for writes."""
        if isinstance(target, IndexExpression):
            access = self._create_access(target, is_write=True)
            if access:
                self._array_accesses.append(access)

    def _analyze_expression(self, expr: Expression, is_write: bool) -> None:
        """Analyze expression for array accesses."""
        if isinstance(expr, IndexExpression):
            access = self._create_access(expr, is_write=is_write)
            if access:
                self._array_accesses.append(access)
            self._analyze_expression(expr.object, is_write=False)
            self._analyze_expression(expr.index, is_write=False)

        elif isinstance(expr, BinaryExpression):
            self._analyze_expression(expr.left, is_write=False)
            self._analyze_expression(expr.right, is_write=False)

        elif isinstance(expr, UnaryExpression):
            self._analyze_expression(expr.operand, is_write=False)

        elif isinstance(expr, CallExpression):
            for arg in expr.arguments:
                self._analyze_expression(arg, is_write=False)

    def _create_access(self, expr: IndexExpression, is_write: bool) -> Optional[ArrayAccess]:
        """Create an ArrayAccess from an IndexExpression."""
        if not isinstance(expr.object, Identifier):
            return None

        array_var = expr.object.name
        depends_on_loop = self._index_depends_on_loop(expr.index)
        stride = self._compute_stride(expr.index)

        return ArrayAccess(
            array_var=array_var,
            indices=[expr.index],
            is_write=is_write,
            loop_var=self._current_loop_var,
            index_depends_on_loop=depends_on_loop,
            stride=stride,
            location=expr.location,
        )

    def _index_depends_on_loop(self, index: Expression) -> bool:
        """Check if index depends on current loop variable."""
        if self._current_loop_var is None:
            return False

        if isinstance(index, Identifier):
            return index.name == self._current_loop_var

        elif isinstance(index, BinaryExpression):
            return (
                self._index_depends_on_loop(index.left) or
                self._index_depends_on_loop(index.right)
            )

        return False

    def _compute_stride(self, index: Expression) -> int:
        """Compute stride of array access relative to loop variable."""
        # Simple case: index is just the loop variable -> stride 1
        if isinstance(index, Identifier):
            if index.name == self._current_loop_var:
                return 1

        # index = loop_var * constant
        if isinstance(index, BinaryExpression):
            if index.operator == BinaryOperator.MUL:
                if isinstance(index.left, Identifier) and isinstance(index.right, IntegerLiteral):
                    if index.left.name == self._current_loop_var:
                        return index.right.value
                if isinstance(index.right, Identifier) and isinstance(index.left, IntegerLiteral):
                    if index.right.name == self._current_loop_var:
                        return index.left.value

        return 1

    def _detect_cache_unfriendly_access(self, loop: ForStatement) -> list[str]:
        """Detect column-major access on row-major arrays, etc."""
        issues: list[str] = []

        # This would require more sophisticated analysis of nested loops
        # and array dimensions

        return issues

    def _suggest_loop_interchange(self, nested_loop) -> Optional[LoopInterchange]:
        """Suggest loop order changes for better cache usage."""
        # Would analyze nested loop access patterns
        return None

    def _suggest_blocking(self, loop: ForStatement) -> Optional[BlockingInfo]:
        """Suggest loop blocking/tiling for cache."""
        # Would analyze data reuse patterns
        return None

    def _generate_analysis(self) -> CacheAnalysis:
        """Generate cache analysis from collected data."""
        if not self._array_accesses:
            return CacheAnalysis(
                access_pattern="none",
                estimated_cache_misses=0,
                suggestions=[],
            )

        # Analyze access patterns
        sequential_count = 0
        strided_count = 0
        total_count = len(self._array_accesses)

        for access in self._array_accesses:
            if access.stride == 1:
                sequential_count += 1
            else:
                strided_count += 1

        # Determine overall pattern
        if sequential_count == total_count:
            pattern = "sequential"
            spatial_locality = 1.0
        elif strided_count > sequential_count:
            pattern = "strided"
            spatial_locality = 0.5
        else:
            pattern = "mixed"
            spatial_locality = 0.7

        # Estimate cache misses (simplified model)
        estimated_misses = strided_count * 10 + sequential_count

        # Generate suggestions
        suggestions: list[str] = []

        if strided_count > 0:
            suggestions.append(
                f"Found {strided_count} strided access(es). Consider loop reordering."
            )

        arrays = set(a.array_var for a in self._array_accesses)

        return CacheAnalysis(
            access_pattern=pattern,
            estimated_cache_misses=estimated_misses,
            suggestions=suggestions,
            loop_var=self._current_loop_var or "",
            arrays_accessed=list(arrays),
            spatial_locality=spatial_locality,
            temporal_locality=1.0,  # Would need more analysis
        )


# =============================================================================
# Memory Layout Optimization
# =============================================================================


class LayoutOptimizer:
    """
    Optimizes memory layout for access patterns.

    Suggests optimal memory layout (C vs Fortran order) based on
    how arrays are accessed in the code.
    """

    def __init__(self) -> None:
        """Initialize the layout optimizer."""
        self._accesses_by_array: dict[str, list[ArrayAccess]] = {}

    def suggest_layout(self, array_uses: list[ArrayAccess]) -> str:
        """
        Suggest optimal memory layout (C vs Fortran order).

        Args:
            array_uses: List of array accesses to analyze

        Returns:
            "C" for row-major, "F" for column-major, "?" for unknown
        """
        if not array_uses:
            return "C"  # Default to C order

        # Group by array
        by_array: dict[str, list[ArrayAccess]] = {}
        for access in array_uses:
            if access.array_var not in by_array:
                by_array[access.array_var] = []
            by_array[access.array_var].append(access)

        # Analyze each array
        suggestions: list[str] = []
        for array_var, accesses in by_array.items():
            layout = self._analyze_access_pattern(accesses)
            suggestions.append(layout)

        # Return most common suggestion
        if not suggestions:
            return "C"

        c_count = suggestions.count("C")
        f_count = suggestions.count("F")

        if c_count >= f_count:
            return "C"
        return "F"

    def _analyze_access_pattern(self, accesses: list[ArrayAccess]) -> str:
        """Determine if row-major or column-major access dominates."""
        # For 2D arrays accessed as arr[i][j]:
        # - If innermost loop varies j (column), use C order (row-major)
        # - If innermost loop varies i (row), use Fortran order (column-major)

        # This is a simplified analysis
        # A full implementation would track multi-dimensional indices

        sequential_count = sum(1 for a in accesses if a.stride == 1)
        total = len(accesses)

        if total == 0:
            return "C"

        if sequential_count / total >= 0.8:
            return "C"  # Good for row-major
        return "?"  # Cannot determine


# =============================================================================
# Temporary Elimination
# =============================================================================


class TemporaryEliminator(BaseASTVisitor):
    """
    Eliminates unnecessary temporary arrays.

    Transformations:
    - temp = a + b; result = temp * c  ->  result = (a + b) * c
    - Or use out= parameter for numpy operations
    """

    def __init__(self) -> None:
        """Initialize the temporary eliminator."""
        self._temp_definitions: dict[str, Expression] = {}
        self._temp_uses: dict[str, int] = {}
        self._elimination_candidates: list[tuple[str, str, str]] = []

    def eliminate(self, func: FunctionDef) -> tuple[FunctionDef, list[str]]:
        """
        Eliminate temporary arrays where possible.

        Args:
            func: Function to optimize

        Returns:
            Tuple of (function, list of elimination messages)
        """
        self._temp_definitions = {}
        self._temp_uses = {}
        self._elimination_candidates = []

        if func.body:
            # First pass: collect definitions and uses
            self._collect_definitions(func.body)

            # Second pass: find elimination candidates
            self._find_candidates()

        messages = [
            f"Can eliminate temporary '{name}': {reason}"
            for name, reason, _ in self._elimination_candidates
        ]

        return func, messages

    def _collect_definitions(self, block: Block) -> None:
        """Collect temporary variable definitions."""
        for stmt in block.statements:
            if isinstance(stmt, LetStatement):
                name = stmt.name
                if name.startswith("_") or name.startswith("tmp") or name.startswith("temp"):
                    if stmt.value:
                        self._temp_definitions[name] = stmt.value
                        self._temp_uses[name] = 0

                # Count uses in value
                if stmt.value:
                    self._count_uses(stmt.value)

            elif isinstance(stmt, AssignmentStatement):
                self._count_uses(stmt.value)
                if isinstance(stmt.target, Identifier):
                    # Check if assigning to temp and then using it once
                    name = stmt.target.name
                    if name.startswith("_") or name.startswith("tmp"):
                        self._temp_definitions[name] = stmt.value
                        self._temp_uses[name] = 0

            elif isinstance(stmt, ForStatement):
                self._count_uses(stmt.iterable)
                self._collect_definitions(stmt.body)

            elif isinstance(stmt, IfStatement):
                self._count_uses(stmt.condition)
                self._collect_definitions(stmt.then_block)
                for cond, blk in stmt.elif_clauses:
                    self._count_uses(cond)
                    self._collect_definitions(blk)
                if stmt.else_block:
                    self._collect_definitions(stmt.else_block)

            elif isinstance(stmt, ReturnStatement) and stmt.value:
                self._count_uses(stmt.value)

            elif isinstance(stmt, ExpressionStatement):
                self._count_uses(stmt.expression)

    def _count_uses(self, expr: Expression) -> None:
        """Count uses of temporary variables."""
        if isinstance(expr, Identifier):
            if expr.name in self._temp_uses:
                self._temp_uses[expr.name] += 1

        elif isinstance(expr, BinaryExpression):
            self._count_uses(expr.left)
            self._count_uses(expr.right)

        elif isinstance(expr, UnaryExpression):
            self._count_uses(expr.operand)

        elif isinstance(expr, CallExpression):
            for arg in expr.arguments:
                self._count_uses(arg)

        elif isinstance(expr, IndexExpression):
            self._count_uses(expr.object)
            self._count_uses(expr.index)

    def _find_candidates(self) -> None:
        """Find temporaries that can be eliminated."""
        for name, use_count in self._temp_uses.items():
            if use_count == 1:
                # Used exactly once - can inline
                self._elimination_candidates.append((
                    name,
                    "used only once, can be inlined",
                    "inline",
                ))
            elif use_count == 0:
                # Never used - dead code
                self._elimination_candidates.append((
                    name,
                    "never used, can be removed",
                    "remove",
                ))


# =============================================================================
# Memory Pool Generation
# =============================================================================


class MemoryPoolGenerator:
    """
    Generates memory pool for repeated allocations.

    For functions called in loops, pre-allocates buffers to avoid
    repeated allocation overhead.
    """

    def __init__(self) -> None:
        """Initialize the memory pool generator."""
        self._allocation_analyzer = AllocationAnalyzer()

    def generate(self, func: FunctionDef) -> str:
        """
        Generate memory pool initialization code.

        Args:
            func: Function to generate pool for

        Returns:
            Generated Python code for the memory pool class
        """
        allocations = self._allocation_analyzer.analyze(func)

        if not allocations:
            return ""

        # Group allocations by type
        type_groups: dict[str, list[AllocationInfo]] = {}
        for alloc in allocations:
            key = alloc.element_type
            if key not in type_groups:
                type_groups[key] = []
            type_groups[key].append(alloc)

        # Generate pool class
        func_name = func.name
        lines = [
            f"class _{func_name}_MemPool:",
            f"    '''Memory pool for {func_name} function.'''",
            "",
            "    def __init__(self, n: int) -> None:",
            "        '''Initialize pool with size n.'''",
        ]

        buffer_idx = 0
        for elem_type, allocs in type_groups.items():
            dtype = "np.float64" if elem_type == "Float" else "np.int64"
            for alloc in allocs:
                lines.append(
                    f"        self.{alloc.variable} = np.empty(n, dtype={dtype})"
                )
                buffer_idx += 1

        lines.append("")
        lines.append("    def get_buffers(self):")
        lines.append("        '''Return all buffer references.'''")
        buf_list = ", ".join(f"self.{a.variable}" for a in allocations)
        lines.append(f"        return ({buf_list})")

        return "\n".join(lines)


# =============================================================================
# Memory Report
# =============================================================================


@dataclass
class MemoryReport:
    """
    Memory optimization report.

    Aggregates all memory optimization analysis results into a
    comprehensive report.
    """

    total_allocations: int = 0
    eliminated_allocations: int = 0
    reused_buffers: int = 0
    estimated_memory_saved: int = 0
    cache_improvements: list[str] = field(default_factory=list)
    inplace_candidates: list[InPlaceCandidate] = field(default_factory=list)
    layout_suggestions: dict[str, str] = field(default_factory=dict)
    buffer_reuse_groups: list[list[str]] = field(default_factory=list)
    temporary_eliminations: list[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Format report as human-readable string."""
        lines = [
            "=" * 60,
            "Memory Optimization Report",
            "=" * 60,
            "",
            f"Total Allocations Found: {self.total_allocations}",
            f"Eliminated Allocations:  {self.eliminated_allocations}",
            f"Reused Buffers:          {self.reused_buffers}",
            f"Estimated Memory Saved:  {self.estimated_memory_saved} bytes",
            "",
        ]

        if self.buffer_reuse_groups:
            lines.append("Buffer Reuse Groups:")
            for i, group in enumerate(self.buffer_reuse_groups):
                lines.append(f"  Group {i + 1}: {', '.join(group)}")
            lines.append("")

        if self.inplace_candidates:
            lines.append("In-Place Operation Candidates:")
            for candidate in self.inplace_candidates:
                lines.append(f"  - {candidate.variable}: {candidate.suggested_code}")
            lines.append("")

        if self.temporary_eliminations:
            lines.append("Temporary Eliminations:")
            for elim in self.temporary_eliminations:
                lines.append(f"  - {elim}")
            lines.append("")

        if self.cache_improvements:
            lines.append("Cache Improvement Suggestions:")
            for suggestion in self.cache_improvements:
                lines.append(f"  - {suggestion}")
            lines.append("")

        if self.layout_suggestions:
            lines.append("Layout Suggestions:")
            for array, layout in self.layout_suggestions.items():
                order = "row-major (C)" if layout == "C" else "column-major (F)"
                lines.append(f"  - {array}: use {order}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Main Memory Optimizer
# =============================================================================


class MemoryOptimizer:
    """
    Main memory optimization pass.

    Coordinates all memory optimization analyses and transformations.
    """

    def __init__(self) -> None:
        """Initialize the memory optimizer."""
        self._allocation_analyzer = AllocationAnalyzer()
        self._buffer_reuse_optimizer = BufferReuseOptimizer()
        self._inplace_optimizer = InPlaceOptimizer()
        self._cache_optimizer = CacheOptimizer()
        self._layout_optimizer = LayoutOptimizer()
        self._temp_eliminator = TemporaryEliminator()
        self._pool_generator = MemoryPoolGenerator()

    def optimize(self, func: FunctionDef) -> tuple[FunctionDef, MemoryReport]:
        """
        Run all memory optimizations on a function.

        Args:
            func: Function to optimize

        Returns:
            Tuple of (optimized function, memory report)
        """
        report = MemoryReport()

        # 1. Allocation Analysis
        allocations = self._allocation_analyzer.analyze(func)
        report.total_allocations = len(allocations)

        # 2. Buffer Reuse
        _, reuse_messages = self._buffer_reuse_optimizer.optimize(func)
        report.reused_buffers = len([m for m in reuse_messages if "can share" in m])

        # 3. In-Place Detection
        _, inplace_candidates = self._inplace_optimizer.optimize(func)
        report.inplace_candidates = inplace_candidates

        # 4. Cache Analysis
        cache_analysis = self._cache_optimizer.analyze(func)
        report.cache_improvements = cache_analysis.suggestions

        # 5. Temporary Elimination
        _, temp_messages = self._temp_eliminator.eliminate(func)
        report.temporary_eliminations = temp_messages
        report.eliminated_allocations = len(temp_messages)

        # 6. Estimate memory savings
        report.estimated_memory_saved = self._estimate_savings(
            allocations, report.reused_buffers, report.eliminated_allocations
        )

        # Note: Actual AST transformation would happen here
        # For now, return original function with analysis report
        return func, report

    def _estimate_savings(
        self,
        allocations: list[AllocationInfo],
        reused: int,
        eliminated: int,
    ) -> int:
        """Estimate memory savings in bytes."""
        # Rough estimate: 8 bytes per float element
        total_savings = 0

        for alloc in allocations:
            if alloc.estimated_size:
                # Assume Float64
                alloc_bytes = alloc.estimated_size * 8
                total_savings += alloc_bytes

        # Adjust for reuse and elimination
        if allocations:
            avg_size = total_savings / len(allocations)
            total_savings = int(avg_size * (reused + eliminated))

        return total_savings

    def analyze_function(self, func: FunctionDef) -> MemoryReport:
        """
        Analyze function without transformation.

        Args:
            func: Function to analyze

        Returns:
            Memory optimization report
        """
        _, report = self.optimize(func)
        return report

    def analyze_program(self, program) -> dict[str, MemoryReport]:
        """
        Analyze all functions in a program.

        Args:
            program: Program AST

        Returns:
            Dictionary mapping function names to their reports
        """
        from mathviz.compiler.ast_nodes import Program

        reports: dict[str, MemoryReport] = {}

        if isinstance(program, Program):
            for stmt in program.statements:
                if isinstance(stmt, FunctionDef):
                    reports[stmt.name] = self.analyze_function(stmt)

        return reports


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_memory(func: FunctionDef) -> MemoryReport:
    """
    Convenience function to analyze memory usage of a function.

    Args:
        func: Function to analyze

    Returns:
        MemoryReport with analysis results
    """
    optimizer = MemoryOptimizer()
    return optimizer.analyze_function(func)


def find_allocations(func: FunctionDef) -> list[AllocationInfo]:
    """
    Convenience function to find all allocations in a function.

    Args:
        func: Function to analyze

    Returns:
        List of AllocationInfo for each allocation
    """
    analyzer = AllocationAnalyzer()
    return analyzer.analyze(func)


def suggest_buffer_reuse(func: FunctionDef) -> list[str]:
    """
    Convenience function to get buffer reuse suggestions.

    Args:
        func: Function to analyze

    Returns:
        List of suggestion messages
    """
    optimizer = BufferReuseOptimizer()
    _, messages = optimizer.optimize(func)
    return messages


def find_inplace_candidates(func: FunctionDef) -> list[InPlaceCandidate]:
    """
    Convenience function to find in-place operation candidates.

    Args:
        func: Function to analyze

    Returns:
        List of InPlaceCandidate objects
    """
    optimizer = InPlaceOptimizer()
    _, candidates = optimizer.optimize(func)
    return candidates


def analyze_cache(func: FunctionDef) -> CacheAnalysis:
    """
    Convenience function to analyze cache behavior.

    Args:
        func: Function to analyze

    Returns:
        CacheAnalysis with detected patterns and suggestions
    """
    optimizer = CacheOptimizer()
    return optimizer.analyze(func)


def generate_memory_pool(func: FunctionDef) -> str:
    """
    Convenience function to generate memory pool code.

    Args:
        func: Function to generate pool for

    Returns:
        Generated Python code for memory pool class
    """
    generator = MemoryPoolGenerator()
    return generator.generate(func)
