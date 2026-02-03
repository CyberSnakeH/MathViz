"""
Algorithmic Complexity Analysis for MathViz AST.

This module provides static analysis to estimate the time complexity
of functions in MathViz programs. It analyzes loop structures, recursion
patterns, and known algorithmic patterns to determine Big-O notation.

The analysis is heuristic-based and provides estimates that are useful
for educational purposes and optimization hints.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mathviz.compiler.ast_nodes import (
    ASTNode,
    BaseASTVisitor,
    BinaryExpression,
    BinaryOperator,
    Block,
    CallExpression,
    CompoundAssignment,
    ForStatement,
    FunctionDef,
    Identifier,
    MemberAccess,
    Program,
    WhileStatement,
)


class Complexity(Enum):
    """
    Big-O complexity classifications.

    Each complexity class represents a different growth rate of
    algorithmic time complexity as input size increases.
    """

    O_1 = "O(1)"  # Constant time
    O_LOG_N = "O(log n)"  # Logarithmic
    O_N = "O(n)"  # Linear
    O_N_LOG_N = "O(n log n)"  # Linearithmic (common in efficient sorting)
    O_N_SQUARED = "O(n\u00b2)"  # Quadratic (nested loops)
    O_N_CUBED = "O(n\u00b3)"  # Cubic (triple nested loops)
    O_2_N = "O(2^n)"  # Exponential (recursive branching)
    O_N_FACTORIAL = "O(n!)"  # Factorial (permutations)
    UNKNOWN = "Unknown"  # Cannot determine complexity

    def __lt__(self, other: "Complexity") -> bool:
        """Compare complexities by growth rate."""
        order = list(Complexity)
        return order.index(self) < order.index(other)

    def __le__(self, other: "Complexity") -> bool:
        return self == other or self < other

    @classmethod
    def from_loop_depth(cls, depth: int) -> "Complexity":
        """Determine complexity from nested loop depth."""
        if depth == 0:
            return cls.O_1
        if depth == 1:
            return cls.O_N
        if depth == 2:
            return cls.O_N_SQUARED
        if depth == 3:
            return cls.O_N_CUBED
        # For deeper nesting, we classify as exponential for practical purposes
        return cls.O_2_N

    @classmethod
    def combine(cls, *complexities: "Complexity") -> "Complexity":
        """
        Combine multiple complexities (sequential operations).

        The resulting complexity is the maximum of all inputs.
        """
        if not complexities:
            return cls.O_1
        # Filter out UNKNOWN and take the maximum
        known = [c for c in complexities if c != cls.UNKNOWN]
        if not known:
            return cls.UNKNOWN
        return max(known, key=lambda c: list(Complexity).index(c))


# Well-known functions and their complexities
KNOWN_FUNCTION_COMPLEXITIES: dict[str, Complexity] = {
    # Sorting functions (typically O(n log n))
    "sort": Complexity.O_N_LOG_N,
    "sorted": Complexity.O_N_LOG_N,
    "heapsort": Complexity.O_N_LOG_N,
    "mergesort": Complexity.O_N_LOG_N,
    "quicksort": Complexity.O_N_LOG_N,
    # Search functions
    "binary_search": Complexity.O_LOG_N,
    "bsearch": Complexity.O_LOG_N,
    "bisect": Complexity.O_LOG_N,
    "bisect_left": Complexity.O_LOG_N,
    "bisect_right": Complexity.O_LOG_N,
    # Linear operations
    "sum": Complexity.O_N,
    "max": Complexity.O_N,
    "min": Complexity.O_N,
    "len": Complexity.O_1,
    "count": Complexity.O_N,
    "index": Complexity.O_N,
    "find": Complexity.O_N,
    "filter": Complexity.O_N,
    "map": Complexity.O_N,
    "reduce": Complexity.O_N,
    "any": Complexity.O_N,
    "all": Complexity.O_N,
    # Constant time operations
    "append": Complexity.O_1,
    "pop": Complexity.O_1,
    "push": Complexity.O_1,
    "get": Complexity.O_1,
    "set": Complexity.O_1,
    "contains": Complexity.O_1,  # For hash-based structures
    "abs": Complexity.O_1,
    "sqrt": Complexity.O_1,
    "log": Complexity.O_1,
    "sin": Complexity.O_1,
    "cos": Complexity.O_1,
    "tan": Complexity.O_1,
    "exp": Complexity.O_1,
    "floor": Complexity.O_1,
    "ceil": Complexity.O_1,
    "round": Complexity.O_1,
    "print": Complexity.O_1,
    "println": Complexity.O_1,
}


@dataclass
class ComplexityInfo:
    """
    Information about the complexity analysis of a function.

    Attributes:
        complexity: The estimated Big-O complexity class.
        loop_depth: Maximum nesting depth of loops in the function.
        has_recursion: Whether the function contains recursive calls.
        recursive_calls: Number of recursive calls per function invocation.
        explanation: Human-readable explanation of the complexity analysis.
    """

    complexity: Complexity
    loop_depth: int
    has_recursion: bool
    recursive_calls: int
    explanation: str

    def __str__(self) -> str:
        parts = [f"Complexity: {self.complexity.value}"]
        if self.loop_depth > 0:
            parts.append(f"Loop depth: {self.loop_depth}")
        if self.has_recursion:
            parts.append(f"Recursive calls: {self.recursive_calls}")
        parts.append(self.explanation)
        return " | ".join(parts)


@dataclass
class _LoopAnalysisResult:
    """Internal result from analyzing a loop structure."""

    depth: int = 0
    has_halving: bool = False
    has_logarithmic_pattern: bool = False


@dataclass
class _RecursionAnalysisResult:
    """Internal result from analyzing recursion patterns."""

    is_recursive: bool = False
    call_count: int = 0
    has_linear_reduction: bool = False  # Reduces by n-1 or similar
    has_halving_reduction: bool = False  # Reduces by n/2


class _LoopDepthVisitor(BaseASTVisitor):
    """
    Visitor to calculate maximum loop nesting depth.

    This visitor traverses the AST and tracks the depth of nested
    for and while loops to determine the loop complexity.
    """

    def __init__(self) -> None:
        self._current_depth: int = 0
        self._max_depth: int = 0
        self._has_halving_loop: bool = False
        self._has_logarithmic_pattern: bool = False

    @property
    def result(self) -> _LoopAnalysisResult:
        return _LoopAnalysisResult(
            depth=self._max_depth,
            has_halving=self._has_halving_loop,
            has_logarithmic_pattern=self._has_logarithmic_pattern,
        )

    def visit_for_statement(self, node: ForStatement) -> Any:
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)

        # Check the iterable for logarithmic patterns
        self._check_logarithmic_iterable(node.iterable)

        # Visit body to find nested loops
        self.visit(node.body)

        self._current_depth -= 1

    def visit_while_statement(self, node: WhileStatement) -> Any:
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)

        # Check for halving patterns in the while body
        self._check_halving_in_block(node.body)

        # Visit body to find nested loops
        self.visit(node.body)

        self._current_depth -= 1

    def _check_logarithmic_iterable(self, expr: ASTNode) -> None:
        """Check if the loop iterates over something that suggests O(log n)."""
        # Example: range with halving step, or iterating over log-sized collections
        # This is a simplified heuristic
        pass

    def _check_halving_in_block(self, block: Block) -> None:
        """Check if the block contains halving operations (i /= 2, i >>= 1, etc.)."""
        for stmt in block.statements:
            if isinstance(stmt, CompoundAssignment):
                if stmt.operator == BinaryOperator.DIV:
                    # Check if dividing by 2
                    if self._is_constant_2(stmt.value):
                        self._has_halving_loop = True
                        self._has_logarithmic_pattern = True
            # Also check for explicit assignment like i = i / 2
            # This would require deeper analysis of AssignmentStatement

    def _is_constant_2(self, expr: ASTNode) -> bool:
        """Check if expression is the constant 2."""
        from mathviz.compiler.ast_nodes import IntegerLiteral

        return isinstance(expr, IntegerLiteral) and expr.value == 2


class _RecursionVisitor(BaseASTVisitor):
    """
    Visitor to detect and analyze recursive function calls.

    This visitor looks for calls to a specific function name within
    a function body to identify recursive patterns.
    """

    def __init__(self, function_name: str) -> None:
        self._function_name = function_name
        self._recursive_call_count: int = 0
        self._argument_patterns: list[tuple[str, str]] = []  # (arg_name, pattern)

    @property
    def result(self) -> _RecursionAnalysisResult:
        has_linear = any(p[1] == "linear" for p in self._argument_patterns)
        has_halving = any(p[1] == "halving" for p in self._argument_patterns)

        return _RecursionAnalysisResult(
            is_recursive=self._recursive_call_count > 0,
            call_count=self._recursive_call_count,
            has_linear_reduction=has_linear,
            has_halving_reduction=has_halving,
        )

    def visit_call_expression(self, node: CallExpression) -> Any:
        # Check if this is a recursive call
        callee_name = self._get_callee_name(node.callee)
        if callee_name == self._function_name:
            self._recursive_call_count += 1
            self._analyze_recursive_arguments(node)

        # Continue visiting arguments in case of nested calls
        for arg in node.arguments:
            self.visit(arg)

    def _get_callee_name(self, callee: ASTNode) -> str | None:
        """Extract the function name from a callee expression."""
        if isinstance(callee, Identifier):
            return callee.name
        if isinstance(callee, MemberAccess):
            return callee.member
        return None

    def _analyze_recursive_arguments(self, call: CallExpression) -> None:
        """Analyze how arguments change in recursive calls."""
        for arg in call.arguments:
            pattern = self._detect_argument_pattern(arg)
            if pattern:
                self._argument_patterns.append(pattern)

    def _detect_argument_pattern(self, arg: ASTNode) -> tuple[str, str] | None:
        """Detect the pattern of argument transformation in recursion."""
        if isinstance(arg, BinaryExpression):
            # Check for n - 1 pattern (linear recursion)
            if arg.operator == BinaryOperator.SUB and isinstance(arg.left, Identifier):
                from mathviz.compiler.ast_nodes import IntegerLiteral

                if isinstance(arg.right, IntegerLiteral) and arg.right.value == 1:
                    return (arg.left.name, "linear")
            # Check for n / 2 pattern (logarithmic recursion)
            if arg.operator in (BinaryOperator.DIV, BinaryOperator.FLOOR_DIV):
                if isinstance(arg.left, Identifier):
                    from mathviz.compiler.ast_nodes import IntegerLiteral

                    if isinstance(arg.right, IntegerLiteral) and arg.right.value == 2:
                        return (arg.left.name, "halving")
        return None


class _FunctionCallVisitor(BaseASTVisitor):
    """
    Visitor to detect calls to known functions with defined complexities.

    This helps identify when sorting or other well-known algorithms
    are used within a function.
    """

    def __init__(self) -> None:
        self._known_call_complexities: list[Complexity] = []

    @property
    def max_known_complexity(self) -> Complexity:
        if not self._known_call_complexities:
            return Complexity.O_1
        return max(self._known_call_complexities, key=lambda c: list(Complexity).index(c))

    def visit_call_expression(self, node: CallExpression) -> Any:
        callee_name = self._get_callee_name(node.callee)
        if callee_name and callee_name.lower() in KNOWN_FUNCTION_COMPLEXITIES:
            complexity = KNOWN_FUNCTION_COMPLEXITIES[callee_name.lower()]
            self._known_call_complexities.append(complexity)

        # Continue visiting
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)

    def _get_callee_name(self, callee: ASTNode) -> str | None:
        """Extract the function name from a callee expression."""
        if isinstance(callee, Identifier):
            return callee.name
        if isinstance(callee, MemberAccess):
            return callee.member
        return None


class ComplexityAnalyzer(BaseASTVisitor):
    """
    Static analyzer for estimating algorithmic complexity of MathViz functions.

    This analyzer uses several heuristics to estimate Big-O complexity:

    1. Loop depth analysis: Counts nested loops to estimate polynomial complexity
    2. Halving pattern detection: Identifies O(log n) patterns like i /= 2
    3. Recursion analysis: Detects recursive calls and their patterns
    4. Known function lookup: Recognizes sorting and other standard algorithms

    Example usage:
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_function(my_function_ast)
        print(f"Complexity: {result.complexity.value}")

    Note:
        This analysis is heuristic-based and provides estimates. It may not
        accurately capture all complexity characteristics, especially for
        complex algorithms with multiple phases or amortized complexity.
    """

    def __init__(self) -> None:
        self._function_complexities: dict[str, ComplexityInfo] = {}

    def analyze_function(self, func: FunctionDef) -> ComplexityInfo:
        """
        Analyze the complexity of a single function.

        Args:
            func: The FunctionDef AST node to analyze.

        Returns:
            ComplexityInfo containing the estimated complexity and analysis details.
        """
        if func.body is None:
            return ComplexityInfo(
                complexity=Complexity.O_1,
                loop_depth=0,
                has_recursion=False,
                recursive_calls=0,
                explanation="Empty function body has constant time complexity.",
            )

        # Analyze loop depth
        loop_visitor = _LoopDepthVisitor()
        loop_visitor.visit(func.body)
        loop_result = loop_visitor.result

        # Analyze recursion
        recursion_visitor = _RecursionVisitor(func.name)
        recursion_visitor.visit(func.body)
        recursion_result = recursion_visitor.result

        # Analyze known function calls
        call_visitor = _FunctionCallVisitor()
        call_visitor.visit(func.body)
        known_call_complexity = call_visitor.max_known_complexity

        # Determine final complexity using heuristics
        complexity, explanation = self._determine_complexity(
            loop_result, recursion_result, known_call_complexity, func.name
        )

        result = ComplexityInfo(
            complexity=complexity,
            loop_depth=loop_result.depth,
            has_recursion=recursion_result.is_recursive,
            recursive_calls=recursion_result.call_count,
            explanation=explanation,
        )

        self._function_complexities[func.name] = result
        return result

    def analyze_program(self, program: Program) -> dict[str, ComplexityInfo]:
        """
        Analyze complexity of all functions in a program.

        Args:
            program: The Program AST node containing all statements.

        Returns:
            Dictionary mapping function names to their ComplexityInfo.
        """
        self._function_complexities.clear()

        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                self.analyze_function(stmt)

        return self._function_complexities.copy()

    def _determine_complexity(
        self,
        loop_result: _LoopAnalysisResult,
        recursion_result: _RecursionAnalysisResult,
        known_call_complexity: Complexity,
        function_name: str,
    ) -> tuple[Complexity, str]:
        """
        Determine the overall complexity using all analysis results.

        This method applies the heuristics to combine loop, recursion,
        and known function complexities.
        """
        explanations: list[str] = []
        candidates: list[Complexity] = []

        # Handle recursion first (can dominate complexity)
        if recursion_result.is_recursive:
            rec_complexity = self._complexity_from_recursion(recursion_result)
            candidates.append(rec_complexity)
            explanations.append(self._explain_recursion(recursion_result, function_name))

        # Handle loops
        if loop_result.depth > 0:
            loop_complexity = self._complexity_from_loops(loop_result)
            candidates.append(loop_complexity)
            explanations.append(self._explain_loops(loop_result))

        # Handle known function calls
        if known_call_complexity != Complexity.O_1:
            candidates.append(known_call_complexity)
            explanations.append(
                f"Contains calls to functions with {known_call_complexity.value} complexity."
            )

        # If no complexity factors found, it's O(1)
        if not candidates:
            return Complexity.O_1, "No loops or recursion detected; constant time operations only."

        # Combine all complexities - take the worst case
        final_complexity = Complexity.combine(*candidates)

        # For recursion within loops, multiply complexities (simplified)
        if recursion_result.is_recursive and loop_result.depth > 0:
            final_complexity = self._multiply_complexities(
                self._complexity_from_recursion(recursion_result),
                self._complexity_from_loops(loop_result),
            )
            explanations.append("Recursion inside loops compounds the complexity.")

        return final_complexity, " ".join(explanations)

    def _complexity_from_loops(self, result: _LoopAnalysisResult) -> Complexity:
        """Determine complexity from loop analysis."""
        if result.has_halving or result.has_logarithmic_pattern:
            if result.depth == 1:
                return Complexity.O_LOG_N
            # Nested logarithmic loops
            return Complexity.O_N_LOG_N

        return Complexity.from_loop_depth(result.depth)

    def _complexity_from_recursion(self, result: _RecursionAnalysisResult) -> Complexity:
        """Determine complexity from recursion analysis."""
        if not result.is_recursive:
            return Complexity.O_1

        # Single recursive call with linear reduction -> O(n)
        if result.call_count == 1:
            if result.has_linear_reduction:
                return Complexity.O_N
            if result.has_halving_reduction:
                return Complexity.O_LOG_N
            # Unknown reduction pattern, assume O(n)
            return Complexity.O_N

        # Two recursive calls (like fibonacci) -> O(2^n)
        if result.call_count == 2:
            if result.has_linear_reduction:
                return Complexity.O_2_N
            if result.has_halving_reduction:
                # Divide and conquer with two calls -> O(n log n) typically
                return Complexity.O_N_LOG_N
            return Complexity.O_2_N

        # More than two recursive calls -> at least exponential
        if result.call_count > 2:
            return Complexity.O_2_N

        return Complexity.UNKNOWN

    def _explain_loops(self, result: _LoopAnalysisResult) -> str:
        """Generate explanation for loop complexity."""
        if result.has_halving or result.has_logarithmic_pattern:
            return f"Loop with halving pattern (depth {result.depth}) indicates logarithmic time."

        if result.depth == 1:
            return "Single loop iterating over input indicates linear time."
        if result.depth == 2:
            return "Nested loops (depth 2) indicate quadratic time."
        if result.depth == 3:
            return "Triple nested loops (depth 3) indicate cubic time."

        return f"Loop depth of {result.depth} indicates polynomial time."

    def _explain_recursion(self, result: _RecursionAnalysisResult, function_name: str) -> str:
        """Generate explanation for recursion complexity."""
        if result.call_count == 1:
            if result.has_linear_reduction:
                return f"Linear recursion in '{function_name}' with n-1 reduction."
            if result.has_halving_reduction:
                return f"Logarithmic recursion in '{function_name}' with n/2 reduction."
            return f"Single recursive call in '{function_name}'."

        if result.call_count == 2:
            if result.has_linear_reduction:
                return (
                    f"Binary recursion in '{function_name}' (like Fibonacci) "
                    f"with {result.call_count} calls per invocation indicates exponential time."
                )
            if result.has_halving_reduction:
                return (
                    f"Divide-and-conquer pattern in '{function_name}' "
                    f"with {result.call_count} calls per invocation."
                )

        return (
            f"Multiple ({result.call_count}) recursive calls in '{function_name}' "
            "suggest exponential complexity."
        )

    def _multiply_complexities(self, c1: Complexity, c2: Complexity) -> Complexity:
        """
        Multiply two complexities (for nested patterns).

        This is a simplified heuristic for cases like recursion inside loops.
        """
        # Simplified multiplication rules
        if c1 == Complexity.O_1:
            return c2
        if c2 == Complexity.O_1:
            return c1

        # O(n) * O(n) = O(n^2)
        if c1 == Complexity.O_N and c2 == Complexity.O_N:
            return Complexity.O_N_SQUARED

        # O(n) * O(log n) = O(n log n)
        if (c1 == Complexity.O_N and c2 == Complexity.O_LOG_N) or (
            c1 == Complexity.O_LOG_N and c2 == Complexity.O_N
        ):
            return Complexity.O_N_LOG_N

        # O(n) * O(n^2) = O(n^3)
        if (c1 == Complexity.O_N and c2 == Complexity.O_N_SQUARED) or (
            c1 == Complexity.O_N_SQUARED and c2 == Complexity.O_N
        ):
            return Complexity.O_N_CUBED

        # For other combinations, return the higher complexity
        return Complexity.combine(c1, c2)


def analyze_complexity(program: Program) -> dict[str, ComplexityInfo]:
    """
    Convenience function to analyze all functions in a program.

    Args:
        program: The Program AST node to analyze.

    Returns:
        Dictionary mapping function names to their ComplexityInfo.

    Example:
        from mathviz.compiler import Parser
        from mathviz.compiler.complexity_analyzer import analyze_complexity

        parser = Parser(tokens)
        ast = parser.parse()
        results = analyze_complexity(ast)

        for name, info in results.items():
            print(f"{name}: {info.complexity.value}")
    """
    analyzer = ComplexityAnalyzer()
    return analyzer.analyze_program(program)
