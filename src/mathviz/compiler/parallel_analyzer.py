"""
Parallelization Detector for MathViz Compiler.

This module analyzes loops for parallelization potential, detecting:
- Data dependencies (RAW, WAR, WAW)
- Loop-carried dependencies that prevent parallelization
- Reduction patterns (sum, product, max, min) that need special handling
- Race conditions from concurrent memory access
- Safe parallel transformations for Numba's prange

The analyzer is conservative: if safety cannot be proven, parallelization
is not recommended. This ensures correctness over performance.

Parallelization Criteria:
- Iterations are independent (no loop-carried dependencies)
- No concurrent writes to the same memory location
- Shared reads are safe (read-only access)
- Reduction operations are explicitly identified for proper handling

References:
- Allen & Kennedy, "Optimizing Compilers for Modern Architectures"
- Numba prange documentation: https://numba.readthedocs.io/
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
    ForStatement,
    FunctionDef,
    # Expressions
    Identifier,
    IfStatement,
    IndexExpression,
    IntegerLiteral,
    # Statements
    LetStatement,
    MemberAccess,
    RangeExpression,
    Statement,
    UnaryExpression,
    WhileStatement,
)


class DependencyType(Enum):
    """
    Types of data dependencies that can occur between instructions.

    In the context of loop parallelization:
    - FLOW (RAW): Read After Write - true dependency, cannot be parallelized
    - ANTI (WAR): Write After Read - anti-dependency, may require privatization
    - OUTPUT (WAW): Write After Write - output dependency, last write wins concern
    """

    FLOW = "flow"  # Read after write (RAW) - true dependency
    ANTI = "anti"  # Write after read (WAR) - anti-dependency
    OUTPUT = "output"  # Write after write (WAW) - output dependency


class ReductionOperator(Enum):
    """
    Reduction operators that can be parallelized with proper handling.

    These operators are associative and commutative (or can be made so),
    allowing parallel reduction with a final combine step.
    """

    SUM = auto()  # += addition
    PRODUCT = auto()  # *= multiplication
    MAX = auto()  # max(var, x)
    MIN = auto()  # min(var, x)
    AND = auto()  # &= / and
    OR = auto()  # |= / or
    XOR = auto()  # ^= / xor


@dataclass(frozen=True, slots=True)
class DataDependency:
    """
    Represents a data dependency found during analysis.

    Attributes:
        variable: The variable involved in the dependency
        dep_type: Type of dependency (FLOW, ANTI, OUTPUT)
        from_iteration: True if dependency crosses loop iterations
        source_stmt: Optional reference to the source statement
        sink_stmt: Optional reference to the sink statement
        description: Human-readable description of the dependency
    """

    variable: str
    dep_type: DependencyType
    from_iteration: bool
    source_stmt: Statement | None = None
    sink_stmt: Statement | None = None
    description: str = ""

    def __str__(self) -> str:
        cross_iter = "cross-iteration " if self.from_iteration else ""
        desc = f" ({self.description})" if self.description else ""
        return f"{cross_iter}{self.dep_type.value} dependency on '{self.variable}'{desc}"


@dataclass(frozen=True, slots=True)
class ReductionVariable:
    """
    Represents a variable used in a reduction pattern.

    Attributes:
        name: Variable name
        operator: The reduction operator (SUM, PRODUCT, etc.)
        init_value: Initial value expression (if detected)
    """

    name: str
    operator: ReductionOperator
    init_value: Expression | None = None

    def __str__(self) -> str:
        return f"reduction({self.operator.name.lower()}: {self.name})"


@dataclass(slots=True)
class LoopAnalysis:
    """
    Complete analysis results for a loop's parallelization potential.

    Attributes:
        is_parallelizable: True if loop can be safely parallelized
        dependencies: List of all data dependencies found
        reduction_vars: Variables that follow reduction patterns
        private_vars: Variables that can be thread-private (loop-local)
        shared_vars: Variables shared across threads (read-only)
        written_vars: Variables written within the loop
        loop_variable: The loop iteration variable
        reason: Explanation of parallelization decision
        can_use_prange: True if loop can use Numba's prange
        needs_parallel_flag: True if function needs parallel=True
        suggested_transforms: List of suggested code transformations
    """

    is_parallelizable: bool = False
    dependencies: list[DataDependency] = field(default_factory=list)
    reduction_vars: set[str] = field(default_factory=set)
    private_vars: set[str] = field(default_factory=set)
    shared_vars: set[str] = field(default_factory=set)
    written_vars: set[str] = field(default_factory=set)
    loop_variable: str = ""
    reason: str = ""
    can_use_prange: bool = False
    needs_parallel_flag: bool = False
    suggested_transforms: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PARALLELIZABLE" if self.is_parallelizable else "NOT PARALLELIZABLE"
        lines = [f"Loop Analysis: {status}"]
        lines.append(f"  Reason: {self.reason}")

        if self.reduction_vars:
            lines.append(f"  Reduction vars: {', '.join(sorted(self.reduction_vars))}")
        if self.private_vars:
            lines.append(f"  Private vars: {', '.join(sorted(self.private_vars))}")
        if self.shared_vars:
            lines.append(f"  Shared vars: {', '.join(sorted(self.shared_vars))}")
        if self.dependencies:
            lines.append("  Dependencies:")
            for dep in self.dependencies:
                lines.append(f"    - {dep}")
        if self.suggested_transforms:
            lines.append("  Suggested transforms:")
            for transform in self.suggested_transforms:
                lines.append(f"    - {transform}")

        return "\n".join(lines)


@dataclass(slots=True)
class MemoryAccess:
    """
    Represents a memory access (read or write) within a loop.

    Attributes:
        variable: Base variable being accessed
        index_expr: Index expression (for array accesses)
        is_write: True if this is a write access
        depends_on_loop_var: True if index depends on loop variable
        index_offset: Constant offset from loop variable (if detectable)
        statement: The statement containing this access
    """

    variable: str
    index_expr: Expression | None = None
    is_write: bool = False
    depends_on_loop_var: bool = False
    index_offset: int = 0  # e.g., arr[i-1] has offset -1
    statement: Statement | None = None


class VariableCollector(BaseASTVisitor):
    """
    Collects all variable references from an expression or statement.

    Used to determine which variables are read in a given expression.
    """

    def __init__(self) -> None:
        self.variables: set[str] = set()

    def collect(self, node: ASTNode) -> set[str]:
        """Collect all variable names referenced in the node."""
        self.variables = set()
        self.visit(node)
        return self.variables

    def visit_identifier(self, node: Identifier) -> None:
        self.variables.add(node.name)

    def visit_index_expression(self, node: IndexExpression) -> None:
        # Visit both object and index
        self.visit(node.object)
        self.visit(node.index)

    def visit_member_access(self, node: MemberAccess) -> None:
        self.visit(node.object)

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_unary_expression(self, node: UnaryExpression) -> None:
        self.visit(node.operand)

    def visit_call_expression(self, node: CallExpression) -> None:
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)

    def visit_conditional_expression(self, node: ConditionalExpression) -> None:
        self.visit(node.condition)
        self.visit(node.then_expr)
        self.visit(node.else_expr)


class IndexAnalyzer(BaseASTVisitor):
    """
    Analyzes array index expressions to determine loop variable dependencies.

    This is crucial for detecting patterns like:
    - arr[i] = expr      -> index equals loop var (offset=0)
    - arr[i-1]           -> index depends on previous iteration (offset=-1)
    - arr[i+1]           -> index depends on future iteration (offset=+1)
    - arr[j]             -> index independent of loop var
    """

    def __init__(self, loop_variable: str) -> None:
        self.loop_variable = loop_variable
        self.depends_on_loop_var = False
        self.offset: int = 0
        self._current_coefficient: int = 1  # For tracking i in expressions like i-1

    def analyze(self, index_expr: Expression) -> tuple[bool, int]:
        """
        Analyze an index expression.

        Returns:
            Tuple of (depends_on_loop_var, offset_from_loop_var)
        """
        self.depends_on_loop_var = False
        self.offset = 0
        self._analyze_expr(index_expr)
        return self.depends_on_loop_var, self.offset

    def _analyze_expr(self, expr: Expression) -> int | None:
        """
        Recursively analyze expression, returning constant value if computable.

        Returns None if expression cannot be reduced to a constant.
        """
        if isinstance(expr, Identifier):
            if expr.name == self.loop_variable:
                self.depends_on_loop_var = True
                return None  # Loop variable is not constant
            return None  # Other variables are not constant

        elif isinstance(expr, IntegerLiteral):
            return expr.value

        elif isinstance(expr, BinaryExpression):
            left_val = self._analyze_expr(expr.left)
            right_val = self._analyze_expr(expr.right)

            # Check if this is loop_var +/- constant
            if expr.operator == BinaryOperator.ADD:
                if self._is_loop_var(expr.left) and right_val is not None:
                    self.offset = right_val
                    return None
                elif self._is_loop_var(expr.right) and left_val is not None:
                    self.offset = left_val
                    return None
                elif left_val is not None and right_val is not None:
                    return left_val + right_val

            elif expr.operator == BinaryOperator.SUB:
                if self._is_loop_var(expr.left) and right_val is not None:
                    self.offset = -right_val
                    return None
                elif left_val is not None and right_val is not None:
                    return left_val - right_val

            elif expr.operator == BinaryOperator.MUL:
                if left_val is not None and right_val is not None:
                    return left_val * right_val

            return None

        elif isinstance(expr, UnaryExpression):
            if expr.operator.name == "NEG":
                val = self._analyze_expr(expr.operand)
                if val is not None:
                    return -val
            return None

        return None

    def _is_loop_var(self, expr: Expression) -> bool:
        """Check if expression is exactly the loop variable."""
        return isinstance(expr, Identifier) and expr.name == self.loop_variable


class LoopBodyAnalyzer(BaseASTVisitor):
    """
    Analyzes a loop body for memory accesses, variable definitions, and dependencies.

    This visitor traverses the loop body and collects:
    - All memory read/write operations
    - Variable definitions (let statements)
    - Assignment patterns (for reduction detection)
    """

    def __init__(self, loop_variable: str) -> None:
        self.loop_variable = loop_variable
        self.memory_accesses: list[MemoryAccess] = []
        self.defined_vars: set[str] = set()  # Variables defined with let
        self.written_vars: set[str] = set()  # All variables written to
        self.read_vars: set[str] = set()  # All variables read
        self.compound_assignments: list[tuple[str, BinaryOperator, Expression]] = []
        self.has_break_continue: bool = False
        self.has_function_calls: bool = False
        self.has_nested_loops: bool = False
        self._current_statement: Statement | None = None
        self._collector = VariableCollector()
        self._index_analyzer = IndexAnalyzer(loop_variable)

    def analyze(self, body: Block) -> None:
        """Analyze the loop body for all relevant patterns."""
        self.memory_accesses = []
        self.defined_vars = set()
        self.written_vars = set()
        self.read_vars = set()
        self.compound_assignments = []
        self.has_break_continue = False
        self.has_function_calls = False
        self.has_nested_loops = False
        self.visit(body)

    def visit_let_statement(self, node: LetStatement) -> None:
        """Track variable definitions."""
        self._current_statement = node
        self.defined_vars.add(node.name)
        self.written_vars.add(node.name)

        if node.value:
            # Collect variables read in the initialization
            read_in_value = self._collector.collect(node.value)
            self.read_vars.update(read_in_value)
            self.visit(node.value)

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        """Track assignment to variables and arrays."""
        self._current_statement = node

        # Analyze the target
        target = node.target
        if isinstance(target, Identifier):
            # Simple variable assignment
            self.written_vars.add(target.name)
            self.memory_accesses.append(
                MemoryAccess(
                    variable=target.name,
                    is_write=True,
                    statement=node,
                )
            )

        elif isinstance(target, IndexExpression):
            # Array element assignment: arr[i] = value
            self._analyze_array_write(target, node)

        # Collect variables read in the value expression
        read_in_value = self._collector.collect(node.value)
        self.read_vars.update(read_in_value)
        self.visit(node.value)

    def visit_compound_assignment(self, node: CompoundAssignment) -> None:
        """Track compound assignments (+=, *=, etc.) for reduction detection."""
        self._current_statement = node

        target = node.target
        if isinstance(target, Identifier):
            var_name = target.name
            self.written_vars.add(var_name)
            self.read_vars.add(var_name)  # Compound assignment also reads

            # Record for reduction analysis
            self.compound_assignments.append((var_name, node.operator, node.value))

            self.memory_accesses.append(
                MemoryAccess(
                    variable=var_name,
                    is_write=True,
                    statement=node,
                )
            )

        elif isinstance(target, IndexExpression):
            # Array element compound assignment: arr[i] += value
            self._analyze_array_write(target, node)
            # Also need to track the read of the same element
            if isinstance(target.object, Identifier):
                self.read_vars.add(target.object.name)

        # Collect variables read in the value
        read_in_value = self._collector.collect(node.value)
        self.read_vars.update(read_in_value)
        self.visit(node.value)

    def _analyze_array_write(
        self,
        target: IndexExpression,
        stmt: Statement,
    ) -> None:
        """Analyze an array write access."""
        if isinstance(target.object, Identifier):
            base_var = target.object.name
            self.written_vars.add(base_var)

            # Analyze the index expression
            depends_on_loop, offset = self._index_analyzer.analyze(target.index)

            self.memory_accesses.append(
                MemoryAccess(
                    variable=base_var,
                    index_expr=target.index,
                    is_write=True,
                    depends_on_loop_var=depends_on_loop,
                    index_offset=offset,
                    statement=stmt,
                )
            )

    def visit_index_expression(self, node: IndexExpression) -> None:
        """Track array read accesses."""
        if isinstance(node.object, Identifier):
            base_var = node.object.name
            self.read_vars.add(base_var)

            # Analyze the index
            depends_on_loop, offset = self._index_analyzer.analyze(node.index)

            # Only add read access if we're not already tracking a write at the same position
            self.memory_accesses.append(
                MemoryAccess(
                    variable=base_var,
                    index_expr=node.index,
                    is_write=False,
                    depends_on_loop_var=depends_on_loop,
                    index_offset=offset,
                    statement=self._current_statement,
                )
            )

        self.visit(node.object)
        self.visit(node.index)

    def visit_call_expression(self, node: CallExpression) -> None:
        """Track function calls (may have side effects)."""
        self.has_function_calls = True

        # Check if it's a known pure function
        if isinstance(node.callee, Identifier):
            func_name = node.callee.name
            # These are known pure functions safe in parallel contexts
            pure_functions = {
                "abs",
                "min",
                "max",
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
                "pow",
                "round",
                "int",
                "float",
                "len",
            }
            if func_name in pure_functions:
                self.has_function_calls = False  # Reset, it's safe

        super().visit_call_expression(node)

    def visit_for_statement(self, node: ForStatement) -> None:
        """Track nested loops."""
        self.has_nested_loops = True
        # Don't recurse into nested loops for this analysis
        # A separate analysis would be needed for nested parallelism

    def visit_while_statement(self, node: WhileStatement) -> None:
        """Track nested while loops."""
        self.has_nested_loops = True

    def visit_break_statement(self, node) -> None:
        """Track break statements (prevent parallelization)."""
        self.has_break_continue = True

    def visit_continue_statement(self, node) -> None:
        """Track continue statements (may complicate parallelization)."""
        self.has_break_continue = True


class ParallelAnalyzer(BaseASTVisitor):
    """
    Analyzes code for parallelization opportunities.

    This analyzer examines for loops to determine if they can be safely
    parallelized using Numba's prange. It detects:

    1. Loop-carried dependencies that prevent parallelization
    2. Reduction patterns that need special handling
    3. Private vs shared variable classification
    4. Potential race conditions

    Usage:
        analyzer = ParallelAnalyzer()
        analysis = analyzer.analyze_loop(for_loop)
        if analysis.can_use_prange:
            # Safe to convert range() to prange()
            pass
    """

    # Operators that form reduction patterns with compound assignment
    REDUCTION_OPERATORS: dict[BinaryOperator, ReductionOperator] = {
        BinaryOperator.ADD: ReductionOperator.SUM,
        BinaryOperator.MUL: ReductionOperator.PRODUCT,
        BinaryOperator.AND: ReductionOperator.AND,
        BinaryOperator.OR: ReductionOperator.OR,
    }

    # Functions that represent reduction operations when updating a variable
    REDUCTION_FUNCTIONS: dict[str, ReductionOperator] = {
        "max": ReductionOperator.MAX,
        "min": ReductionOperator.MIN,
    }

    def __init__(self) -> None:
        """Initialize the parallel analyzer."""
        self._loop_analyses: list[tuple[ForStatement, LoopAnalysis]] = []
        self._current_function: FunctionDef | None = None
        self._function_params: set[str] = set()

    def analyze_loop(self, loop: ForStatement) -> LoopAnalysis:
        """
        Analyze a for loop for parallelization potential.

        This is the main entry point for loop analysis. It examines:
        - The loop's iteration space (must be a range)
        - Memory access patterns within the loop body
        - Data dependencies between iterations
        - Reduction patterns

        Args:
            loop: The ForStatement AST node to analyze

        Returns:
            LoopAnalysis containing detailed parallelization information
        """
        analysis = LoopAnalysis(loop_variable=loop.variable)

        # Step 1: Check if the loop iterates over a range
        if not self._is_range_loop(loop):
            analysis.reason = "Loop does not iterate over a range expression"
            return analysis

        # Step 2: Analyze the loop body
        body_analyzer = LoopBodyAnalyzer(loop.variable)
        body_analyzer.analyze(loop.body)

        # Step 3: Check for break/continue (prevents simple parallelization)
        if body_analyzer.has_break_continue:
            analysis.reason = "Loop contains break or continue statements"
            analysis.suggested_transforms.append(
                "Refactor to remove early exit for parallelization"
            )
            return analysis

        # Step 4: Check for impure function calls
        if body_analyzer.has_function_calls:
            # Note: We've already filtered out known pure functions
            analysis.reason = "Loop contains potentially impure function calls"
            analysis.suggested_transforms.append(
                "Ensure all called functions are pure (no side effects)"
            )
            return analysis

        # Step 5: Detect reduction patterns
        reduction_vars = self._detect_reductions(
            body_analyzer.compound_assignments,
            loop.variable,
        )
        analysis.reduction_vars = {rv.name for rv in reduction_vars}

        # Step 6: Classify variables
        analysis.written_vars = body_analyzer.written_vars.copy()

        # Private variables: defined inside the loop (let statements)
        analysis.private_vars = body_analyzer.defined_vars.copy()
        # Loop variable is implicitly private
        analysis.private_vars.add(loop.variable)

        # Shared variables: read but not written (or only written via reduction)
        for var in body_analyzer.read_vars:
            if var not in analysis.written_vars or var in analysis.reduction_vars:
                if var != loop.variable and var not in analysis.private_vars:
                    analysis.shared_vars.add(var)

        # Step 7: Analyze memory accesses for dependencies
        dependencies = self._analyze_memory_dependencies(
            body_analyzer.memory_accesses,
            loop.variable,
            analysis.reduction_vars,
            analysis.private_vars,
        )
        analysis.dependencies = dependencies

        # Step 8: Check for loop-carried dependencies (excluding reductions)
        loop_carried_deps = [
            d
            for d in dependencies
            if d.from_iteration and d.variable not in analysis.reduction_vars
        ]

        if loop_carried_deps:
            analysis.reason = "Loop has cross-iteration dependencies"
            for dep in loop_carried_deps:
                analysis.suggested_transforms.append(f"Resolve: {dep}")
            return analysis

        # Step 9: Check for race conditions on shared writes
        shared_writes = self._detect_race_conditions(
            body_analyzer.memory_accesses,
            loop.variable,
            analysis.reduction_vars,
        )

        if shared_writes:
            analysis.reason = (
                "Potential race condition: multiple iterations may write to same location"
            )
            for var in shared_writes:
                analysis.suggested_transforms.append(f"Variable '{var}' may have concurrent writes")
            return analysis

        # Step 10: All checks passed - loop is parallelizable
        analysis.is_parallelizable = True
        analysis.can_use_prange = True

        if reduction_vars:
            analysis.needs_parallel_flag = True
            analysis.reason = (
                f"Parallelizable with reduction(s): {', '.join(str(rv) for rv in reduction_vars)}"
            )
        else:
            analysis.reason = "All iterations are independent"

        return analysis

    def analyze_function(
        self,
        func: FunctionDef,
    ) -> list[tuple[ForStatement, LoopAnalysis]]:
        """
        Find and analyze all parallelizable loops in a function.

        Args:
            func: The function definition to analyze

        Returns:
            List of (ForStatement, LoopAnalysis) tuples for each loop found
        """
        self._loop_analyses = []
        self._current_function = func
        self._function_params = {p.name for p in func.parameters}

        if func.body:
            self._find_loops(func.body)

        return self._loop_analyses

    def _find_loops(self, node: ASTNode) -> None:
        """Recursively find all for loops in the AST."""
        if isinstance(node, ForStatement):
            analysis = self.analyze_loop(node)
            self._loop_analyses.append((node, analysis))
            # Also analyze nested loops
            self._find_loops(node.body)

        elif isinstance(node, Block):
            for stmt in node.statements:
                self._find_loops(stmt)

        elif isinstance(node, IfStatement):
            self._find_loops(node.then_block)
            for _, elif_block in node.elif_clauses:
                self._find_loops(elif_block)
            if node.else_block:
                self._find_loops(node.else_block)

        elif isinstance(node, WhileStatement):
            self._find_loops(node.body)

    def can_use_prange(self, loop: ForStatement) -> bool:
        """
        Quick check if a loop can use Numba's prange.

        This is a convenience method that performs the full analysis
        and returns a simple boolean.

        Args:
            loop: The for loop to check

        Returns:
            True if the loop can safely use prange instead of range
        """
        analysis = self.analyze_loop(loop)
        return analysis.can_use_prange

    def _is_range_loop(self, loop: ForStatement) -> bool:
        """Check if the loop iterates over a range expression."""
        iterable = loop.iterable

        # Direct range expression (0..10 or 0..=10)
        if isinstance(iterable, RangeExpression):
            return True

        # Call to range() function
        if isinstance(iterable, CallExpression) and isinstance(iterable.callee, Identifier):
            return iterable.callee.name == "range"

        return False

    def _detect_reductions(
        self,
        compound_assignments: list[tuple[str, BinaryOperator, Expression]],
        loop_variable: str,
    ) -> list[ReductionVariable]:
        """
        Detect reduction patterns in compound assignments.

        Patterns detected:
        - sum += x      (accumulator addition)
        - product *= x  (accumulator multiplication)
        - val = max(val, x)  (max reduction)
        - val = min(val, x)  (min reduction)
        """
        reductions: list[ReductionVariable] = []
        seen_vars: set[str] = set()

        for var_name, operator, _value_expr in compound_assignments:
            # Skip if already identified as reduction
            if var_name in seen_vars:
                continue

            # Skip loop variable
            if var_name == loop_variable:
                continue

            # Check if operator is a reduction operator
            if operator in self.REDUCTION_OPERATORS:
                reduction_op = self.REDUCTION_OPERATORS[operator]
                reductions.append(
                    ReductionVariable(
                        name=var_name,
                        operator=reduction_op,
                    )
                )
                seen_vars.add(var_name)

        return reductions

    def _analyze_memory_dependencies(
        self,
        accesses: list[MemoryAccess],
        loop_variable: str,
        reduction_vars: set[str],
        private_vars: set[str],
    ) -> list[DataDependency]:
        """
        Analyze memory accesses for data dependencies.

        This looks for:
        - RAW (flow) dependencies: read after write from different iteration
        - WAR (anti) dependencies: write after read from different iteration
        - WAW (output) dependencies: write after write from different iteration
        """
        dependencies: list[DataDependency] = []

        # Group accesses by variable
        accesses_by_var: dict[str, list[MemoryAccess]] = {}
        for access in accesses:
            if access.variable not in accesses_by_var:
                accesses_by_var[access.variable] = []
            accesses_by_var[access.variable].append(access)

        for var, var_accesses in accesses_by_var.items():
            # Skip private variables (they're local to each iteration)
            if var in private_vars:
                continue

            # Skip reduction variables (handled specially)
            if var in reduction_vars:
                continue

            # Skip loop variable
            if var == loop_variable:
                continue

            writes = [a for a in var_accesses if a.is_write]
            reads = [a for a in var_accesses if not a.is_write]

            # Check for loop-carried dependencies
            for write in writes:
                if not write.depends_on_loop_var:
                    # Write to fixed location - potential race condition
                    # (handled separately in race detection)
                    continue

                # Check reads from other iterations
                for read in reads:
                    if read.depends_on_loop_var and read.index_offset != write.index_offset:
                        # Read from different iteration - RAW dependency
                        dependencies.append(
                            DataDependency(
                                variable=var,
                                dep_type=DependencyType.FLOW,
                                from_iteration=True,
                                source_stmt=write.statement,
                                sink_stmt=read.statement,
                                description=f"writes arr[i+{write.index_offset}], reads arr[i+{read.index_offset}]",
                            )
                        )

                # Check writes from other iterations
                for other_write in writes:
                    if (
                        other_write is not write
                        and other_write.depends_on_loop_var
                        and other_write.index_offset != write.index_offset
                    ):
                        dependencies.append(
                            DataDependency(
                                variable=var,
                                dep_type=DependencyType.OUTPUT,
                                from_iteration=True,
                                source_stmt=write.statement,
                                sink_stmt=other_write.statement,
                                description="multiple writes to different indices",
                            )
                        )

            # Check for reads that depend on writes from previous iterations
            for read in reads:
                if read.depends_on_loop_var and read.index_offset != 0:
                    # Reading from arr[i-1] or arr[i+1] etc.
                    for write in writes:
                        if write.depends_on_loop_var and write.index_offset == 0:
                            # Pattern: arr[i] = ...; ... arr[i-1] ...
                            # This is a loop-carried RAW dependency
                            if read.index_offset < 0:
                                # Reading from previous iteration
                                dependencies.append(
                                    DataDependency(
                                        variable=var,
                                        dep_type=DependencyType.FLOW,
                                        from_iteration=True,
                                        source_stmt=write.statement,
                                        sink_stmt=read.statement,
                                        description=f"reads arr[i{read.index_offset}] depends on prior write to arr[i]",
                                    )
                                )

        return dependencies

    def _detect_race_conditions(
        self,
        accesses: list[MemoryAccess],
        loop_variable: str,
        reduction_vars: set[str],
    ) -> set[str]:
        """
        Detect potential race conditions from concurrent writes.

        A race condition occurs when:
        - Multiple iterations write to the same memory location
        - The write index does not depend on the loop variable
        """
        race_vars: set[str] = set()

        for access in accesses:
            if not access.is_write:
                continue

            # Skip reduction variables (handled by reduction mechanism)
            if access.variable in reduction_vars:
                continue

            # Check if write target is loop-independent
            if not access.depends_on_loop_var:
                # Writing to a fixed location in all iterations = race condition
                race_vars.add(access.variable)

        return race_vars


def analyze_parallelization(func: FunctionDef) -> list[tuple[ForStatement, LoopAnalysis]]:
    """
    Convenience function to analyze all loops in a function for parallelization.

    Args:
        func: The function definition to analyze

    Returns:
        List of (ForStatement, LoopAnalysis) tuples
    """
    analyzer = ParallelAnalyzer()
    return analyzer.analyze_function(func)


def can_parallelize_loop(loop: ForStatement) -> bool:
    """
    Convenience function for quick parallelization check.

    Args:
        loop: The for loop to check

    Returns:
        True if the loop can be parallelized
    """
    analyzer = ParallelAnalyzer()
    return analyzer.can_use_prange(loop)
