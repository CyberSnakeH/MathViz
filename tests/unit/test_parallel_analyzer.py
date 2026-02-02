"""
Unit tests for the Parallelization Detector module.

Tests cover:
- Data dependency detection (RAW, WAR, WAW)
- Loop-carried dependency identification
- Reduction pattern recognition
- Race condition detection
- Variable classification (private, shared, reduction)
- Edge cases and complex patterns

Note: Since the parser's range expression (0..n) has issues, we construct
AST nodes directly for most tests to test the analyzer in isolation.
"""

import pytest

from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.ast_nodes import (
    ForStatement,
    FunctionDef,
    Block,
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    RangeExpression,
    AssignmentStatement,
    LetStatement,
    CompoundAssignment,
    IndexExpression,
    BinaryExpression,
    BinaryOperator,
    CallExpression,
    IfStatement,
    BreakStatement,
    ContinueStatement,
    Parameter,
    SimpleType,
)
from mathviz.compiler.parallel_analyzer import (
    ParallelAnalyzer,
    LoopAnalysis,
    DataDependency,
    DependencyType,
    ReductionOperator,
    ReductionVariable,
    VariableCollector,
    IndexAnalyzer,
    LoopBodyAnalyzer,
    analyze_parallelization,
    can_parallelize_loop,
)


# ============================================================================
# Helper functions to create AST nodes directly
# ============================================================================

def make_identifier(name: str) -> Identifier:
    """Create an Identifier node."""
    return Identifier(name=name)


def make_int(value: int) -> IntegerLiteral:
    """Create an IntegerLiteral node."""
    return IntegerLiteral(value=value)


def make_index(array: str, index_expr) -> IndexExpression:
    """Create an IndexExpression node."""
    return IndexExpression(
        object=make_identifier(array),
        index=index_expr,
    )


def make_assignment(target, value) -> AssignmentStatement:
    """Create an AssignmentStatement node."""
    return AssignmentStatement(target=target, value=value)


def make_compound_assignment(target, operator: BinaryOperator, value) -> CompoundAssignment:
    """Create a CompoundAssignment node."""
    return CompoundAssignment(target=target, operator=operator, value=value)


def make_let(name: str, value=None) -> LetStatement:
    """Create a LetStatement node."""
    return LetStatement(name=name, value=value)


def make_binary(left, op: BinaryOperator, right) -> BinaryExpression:
    """Create a BinaryExpression node."""
    return BinaryExpression(left=left, operator=op, right=right)


def make_range_loop(
    variable: str,
    start: int,
    end_var: str,
    statements: tuple,
) -> ForStatement:
    """Create a ForStatement with a RangeExpression as iterable."""
    return ForStatement(
        variable=variable,
        iterable=RangeExpression(
            start=IntegerLiteral(value=start),
            end=Identifier(name=end_var),
            inclusive=False,
        ),
        body=Block(statements=statements),
    )


def make_function(
    name: str,
    params: list[tuple[str, str]],
    body_statements: tuple,
) -> FunctionDef:
    """Create a FunctionDef node with parameters and body."""
    parameters = tuple(
        Parameter(name=p_name, type_annotation=SimpleType(name=p_type))
        for p_name, p_type in params
    )
    return FunctionDef(
        name=name,
        parameters=parameters,
        body=Block(statements=body_statements),
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def analyzer():
    """Create a fresh ParallelAnalyzer instance."""
    return ParallelAnalyzer()


# ============================================================================
# Basic Parallelization Tests
# ============================================================================

class TestBasicParallelization:
    """Tests for simple parallelizable loops."""

    def test_independent_iterations_simple(self, analyzer):
        """Test: arr[i] = i * 2 - each iteration writes to unique location."""
        # for i in 0..n { arr[i] = i * 2 }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_identifier("i"),
                        BinaryOperator.MUL,
                        make_int(2),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert analysis.can_use_prange
        assert "independent" in analysis.reason.lower()

    def test_independent_iterations_with_expression(self, analyzer):
        """Test: arr[i] = arr[i] * 2 - reads and writes same index."""
        # for i in 0..n { arr[i] = arr[i] * 2 }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.MUL,
                        make_int(2),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert analysis.can_use_prange

    def test_multiple_arrays_independent(self, analyzer):
        """Test: Multiple arrays, all indexed by loop variable."""
        # for i in 0..n { c[i] = a[i] + b[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("c", make_identifier("i")),
                    value=make_binary(
                        make_index("a", make_identifier("i")),
                        BinaryOperator.ADD,
                        make_index("b", make_identifier("i")),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        # a and b should be shared (read-only), c is written
        assert "a" in analysis.shared_vars
        assert "b" in analysis.shared_vars
        assert "c" in analysis.written_vars


# ============================================================================
# Loop-Carried Dependency Tests
# ============================================================================

class TestLoopCarriedDependencies:
    """Tests for loops with cross-iteration dependencies."""

    def test_raw_dependency_previous_iteration(self, analyzer):
        """Test: arr[i] = arr[i-1] + 1 - reads from previous iteration."""
        # for i in 1..n { arr[i] = arr[i - 1] + 1 }
        loop = make_range_loop(
            variable="i",
            start=1,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_index(
                            "arr",
                            make_binary(
                                make_identifier("i"),
                                BinaryOperator.SUB,
                                make_int(1),
                            ),
                        ),
                        BinaryOperator.ADD,
                        make_int(1),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert not analysis.is_parallelizable
        assert not analysis.can_use_prange

        # Should have a FLOW (RAW) dependency
        flow_deps = [d for d in analysis.dependencies if d.dep_type == DependencyType.FLOW]
        assert len(flow_deps) > 0
        assert flow_deps[0].from_iteration


# ============================================================================
# Reduction Pattern Tests
# ============================================================================

class TestReductionPatterns:
    """Tests for reduction pattern detection."""

    def test_sum_reduction(self, analyzer):
        """Test: sum += arr[i] - addition reduction."""
        # for i in 0..n { sum += arr[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_compound_assignment(
                    target=make_identifier("sum"),
                    operator=BinaryOperator.ADD,
                    value=make_index("arr", make_identifier("i")),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert "sum" in analysis.reduction_vars
        assert analysis.needs_parallel_flag
        assert "reduction" in analysis.reason.lower()

    def test_product_reduction(self, analyzer):
        """Test: product *= arr[i] - multiplication reduction."""
        # for i in 0..n { product *= arr[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_compound_assignment(
                    target=make_identifier("product"),
                    operator=BinaryOperator.MUL,
                    value=make_index("arr", make_identifier("i")),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert "product" in analysis.reduction_vars

    def test_multiple_reductions(self, analyzer):
        """Test: Multiple reduction variables in same loop."""
        # for i in 0..n { sum += arr[i]; sum_sq += arr[i] * arr[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_compound_assignment(
                    target=make_identifier("sum"),
                    operator=BinaryOperator.ADD,
                    value=make_index("arr", make_identifier("i")),
                ),
                make_compound_assignment(
                    target=make_identifier("sum_sq"),
                    operator=BinaryOperator.ADD,
                    value=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.MUL,
                        make_index("arr", make_identifier("i")),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert "sum" in analysis.reduction_vars
        assert "sum_sq" in analysis.reduction_vars


# ============================================================================
# Race Condition Tests
# ============================================================================

class TestRaceConditions:
    """Tests for race condition detection."""

    def test_write_to_fixed_location(self, analyzer):
        """Test: result = arr[i] - all iterations write to same variable."""
        # for i in 0..n { result = arr[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_identifier("result"),
                    value=make_index("arr", make_identifier("i")),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        # This is a race condition - multiple iterations write to 'result'
        assert not analysis.is_parallelizable

    def test_write_to_fixed_array_index(self, analyzer):
        """Test: arr[0] = i - all iterations write to arr[0]."""
        # for i in 0..n { arr[0] = i }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_int(0)),
                    value=make_identifier("i"),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        # Writing to arr[0] in every iteration is a race
        assert not analysis.is_parallelizable


# ============================================================================
# Variable Classification Tests
# ============================================================================

class TestVariableClassification:
    """Tests for private/shared variable classification."""

    def test_loop_local_variable_is_private(self, analyzer):
        """Test: Variables defined inside loop are private."""
        # for i in 0..n { let temp = arr[i] * 2; arr[i] = temp + 1 }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_let(
                    name="temp",
                    value=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.MUL,
                        make_int(2),
                    ),
                ),
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_identifier("temp"),
                        BinaryOperator.ADD,
                        make_int(1),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert "temp" in analysis.private_vars
        assert "i" in analysis.private_vars  # Loop variable is always private

    def test_read_only_variable_is_shared(self, analyzer):
        """Test: Variables only read are shared."""
        # for i in 0..n { arr[i] = arr[i] * factor }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.MUL,
                        make_identifier("factor"),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        # 'factor' is read but not written - should be shared
        assert "factor" in analysis.shared_vars


# ============================================================================
# Control Flow Tests
# ============================================================================

class TestControlFlow:
    """Tests for loops with control flow statements."""

    def test_loop_with_break_not_parallelizable(self, analyzer):
        """Test: Loops with break cannot be parallelized."""
        # for i in 0..n { if arr[i] == target { break } }
        loop = ForStatement(
            variable="i",
            iterable=RangeExpression(
                start=make_int(0),
                end=make_identifier("n"),
            ),
            body=Block(statements=(
                IfStatement(
                    condition=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.EQ,
                        make_identifier("target"),
                    ),
                    then_block=Block(statements=(BreakStatement(),)),
                ),
            )),
        )
        analysis = analyzer.analyze_loop(loop)

        assert not analysis.is_parallelizable
        assert "break" in analysis.reason.lower() or "continue" in analysis.reason.lower()

    def test_loop_with_continue_not_parallelizable(self, analyzer):
        """Test: Loops with continue may not be parallelizable."""
        # for i in 0..n { if arr[i] == 0 { continue }; arr[i] = arr[i] * 2 }
        loop = ForStatement(
            variable="i",
            iterable=RangeExpression(
                start=make_int(0),
                end=make_identifier("n"),
            ),
            body=Block(statements=(
                IfStatement(
                    condition=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.EQ,
                        make_int(0),
                    ),
                    then_block=Block(statements=(ContinueStatement(),)),
                ),
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.MUL,
                        make_int(2),
                    ),
                ),
            )),
        )
        analysis = analyzer.analyze_loop(loop)

        # Continue might prevent simple parallelization
        assert not analysis.is_parallelizable


# ============================================================================
# Function Analysis Tests
# ============================================================================

class TestFunctionAnalysis:
    """Tests for analyzing entire functions."""

    def test_analyze_function_finds_all_loops(self, analyzer):
        """Test: analyze_function finds all loops in a function."""
        # fn multi_loop() { for i in 0..n { a[i] = i }; for j in 0..n { b[j] = j * 2 } }
        func = make_function(
            name="multi_loop",
            params=[("a", "List"), ("b", "List"), ("n", "Int")],
            body_statements=(
                make_range_loop(
                    variable="i",
                    start=0,
                    end_var="n",
                    statements=(
                        make_assignment(
                            target=make_index("a", make_identifier("i")),
                            value=make_identifier("i"),
                        ),
                    ),
                ),
                make_range_loop(
                    variable="j",
                    start=0,
                    end_var="n",
                    statements=(
                        make_assignment(
                            target=make_index("b", make_identifier("j")),
                            value=make_binary(
                                make_identifier("j"),
                                BinaryOperator.MUL,
                                make_int(2),
                            ),
                        ),
                    ),
                ),
            ),
        )
        results = analyzer.analyze_function(func)

        assert len(results) == 2
        assert all(isinstance(loop, ForStatement) for loop, _ in results)
        assert all(isinstance(analysis, LoopAnalysis) for _, analysis in results)

    def test_analyze_nested_loops(self, analyzer):
        """Test: Nested loops are detected."""
        # fn matrix_init() { for i in 0..n { for j in 0..m { ... } } }
        inner_loop = make_range_loop(
            variable="j",
            start=0,
            end_var="m",
            statements=(
                make_assignment(
                    target=IndexExpression(
                        object=make_index("mat", make_identifier("i")),
                        index=make_identifier("j"),
                    ),
                    value=make_binary(
                        make_binary(
                            make_identifier("i"),
                            BinaryOperator.MUL,
                            make_identifier("m"),
                        ),
                        BinaryOperator.ADD,
                        make_identifier("j"),
                    ),
                ),
            ),
        )

        outer_loop = ForStatement(
            variable="i",
            iterable=RangeExpression(
                start=make_int(0),
                end=make_identifier("n"),
            ),
            body=Block(statements=(inner_loop,)),
        )

        func = FunctionDef(
            name="matrix_init",
            parameters=(
                Parameter(name="mat", type_annotation=SimpleType(name="List")),
                Parameter(name="n", type_annotation=SimpleType(name="Int")),
                Parameter(name="m", type_annotation=SimpleType(name="Int")),
            ),
            body=Block(statements=(outer_loop,)),
        )

        results = analyzer.analyze_function(func)

        # Should find both outer and inner loops
        assert len(results) == 2


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_parallelization_function(self):
        """Test: analyze_parallelization convenience function."""
        func = make_function(
            name="simple",
            params=[("arr", "List"), ("n", "Int")],
            body_statements=(
                make_range_loop(
                    variable="i",
                    start=0,
                    end_var="n",
                    statements=(
                        make_assignment(
                            target=make_index("arr", make_identifier("i")),
                            value=make_identifier("i"),
                        ),
                    ),
                ),
            ),
        )
        results = analyze_parallelization(func)

        assert len(results) == 1
        assert results[0][1].is_parallelizable

    def test_can_parallelize_loop_function(self):
        """Test: can_parallelize_loop convenience function."""
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_identifier("i"),
                ),
            ),
        )
        assert can_parallelize_loop(loop)


# ============================================================================
# Helper Class Tests
# ============================================================================

class TestVariableCollector:
    """Tests for VariableCollector helper class."""

    def test_collect_from_identifier(self):
        """Test collecting from simple identifier."""
        collector = VariableCollector()
        node = Identifier(name="x")
        vars_found = collector.collect(node)
        assert vars_found == {"x"}

    def test_collect_from_binary_expression(self):
        """Test collecting from binary expression."""
        collector = VariableCollector()
        node = BinaryExpression(
            left=Identifier(name="a"),
            operator=BinaryOperator.ADD,
            right=Identifier(name="b"),
        )
        vars_found = collector.collect(node)
        assert vars_found == {"a", "b"}

    def test_collect_from_index_expression(self):
        """Test collecting from index expression."""
        collector = VariableCollector()
        node = IndexExpression(
            object=Identifier(name="arr"),
            index=Identifier(name="i"),
        )
        vars_found = collector.collect(node)
        assert vars_found == {"arr", "i"}


class TestIndexAnalyzer:
    """Tests for IndexAnalyzer helper class."""

    def test_simple_loop_variable(self):
        """Test: index is exactly the loop variable."""
        idx_analyzer = IndexAnalyzer(loop_variable="i")
        depends, offset = idx_analyzer.analyze(Identifier(name="i"))
        assert depends
        assert offset == 0

    def test_loop_variable_plus_constant(self):
        """Test: i + 1 pattern."""
        idx_analyzer = IndexAnalyzer(loop_variable="i")
        expr = BinaryExpression(
            left=Identifier(name="i"),
            operator=BinaryOperator.ADD,
            right=IntegerLiteral(value=1),
        )
        depends, offset = idx_analyzer.analyze(expr)
        assert depends
        assert offset == 1

    def test_loop_variable_minus_constant(self):
        """Test: i - 1 pattern."""
        idx_analyzer = IndexAnalyzer(loop_variable="i")
        expr = BinaryExpression(
            left=Identifier(name="i"),
            operator=BinaryOperator.SUB,
            right=IntegerLiteral(value=1),
        )
        depends, offset = idx_analyzer.analyze(expr)
        assert depends
        assert offset == -1

    def test_independent_variable(self):
        """Test: index is independent of loop variable."""
        idx_analyzer = IndexAnalyzer(loop_variable="i")
        depends, offset = idx_analyzer.analyze(Identifier(name="j"))
        assert not depends

    def test_constant_index(self):
        """Test: constant index."""
        idx_analyzer = IndexAnalyzer(loop_variable="i")
        depends, offset = idx_analyzer.analyze(IntegerLiteral(value=5))
        assert not depends


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_loop_body(self, analyzer):
        """Test: Loop with empty body."""
        loop = ForStatement(
            variable="i",
            iterable=RangeExpression(
                start=IntegerLiteral(value=0),
                end=IntegerLiteral(value=10),
            ),
            body=Block(statements=()),
        )
        analysis = analyzer.analyze_loop(loop)

        # Empty loop is trivially parallelizable
        assert analysis.is_parallelizable

    def test_non_range_loop(self, analyzer):
        """Test: Loop over non-range iterable (identifier)."""
        loop = ForStatement(
            variable="item",
            iterable=Identifier(name="items"),
            body=Block(statements=(
                make_let("x", make_identifier("item")),
            )),
        )
        analysis = analyzer.analyze_loop(loop)

        # Only range loops can be converted to prange
        assert not analysis.can_use_prange
        assert "range" in analysis.reason.lower()

    def test_loop_analysis_string_representation(self, analyzer):
        """Test: LoopAnalysis __str__ method."""
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_compound_assignment(
                    target=make_identifier("sum"),
                    operator=BinaryOperator.ADD,
                    value=make_index("arr", make_identifier("i")),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        str_repr = str(analysis)
        assert "PARALLELIZABLE" in str_repr
        assert "sum" in str_repr.lower()

    def test_data_dependency_string_representation(self):
        """Test: DataDependency __str__ method."""
        dep = DataDependency(
            variable="arr",
            dep_type=DependencyType.FLOW,
            from_iteration=True,
            description="writes arr[i], reads arr[i-1]",
        )
        str_repr = str(dep)
        assert "cross-iteration" in str_repr
        assert "flow" in str_repr
        assert "arr" in str_repr


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with real-world patterns."""

    def test_dot_product_pattern(self, analyzer):
        """Test: Dot product - classic reduction pattern."""
        # for i in 0..n { result += a[i] * b[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_compound_assignment(
                    target=make_identifier("result"),
                    operator=BinaryOperator.ADD,
                    value=make_binary(
                        make_index("a", make_identifier("i")),
                        BinaryOperator.MUL,
                        make_index("b", make_identifier("i")),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert "result" in analysis.reduction_vars
        assert analysis.needs_parallel_flag

    def test_saxpy_pattern(self, analyzer):
        """Test: SAXPY (y = a*x + y) - classic parallel pattern."""
        # for i in 0..n { y[i] = a * x[i] + y[i] }
        loop = make_range_loop(
            variable="i",
            start=0,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("y", make_identifier("i")),
                    value=make_binary(
                        make_binary(
                            make_identifier("a"),
                            BinaryOperator.MUL,
                            make_index("x", make_identifier("i")),
                        ),
                        BinaryOperator.ADD,
                        make_index("y", make_identifier("i")),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert analysis.is_parallelizable
        assert analysis.can_use_prange
        assert "a" in analysis.shared_vars

    def test_prefix_sum_not_parallelizable(self, analyzer):
        """Test: Prefix sum - classic non-parallelizable pattern."""
        # for i in 1..n { arr[i] = arr[i] + arr[i-1] }
        loop = make_range_loop(
            variable="i",
            start=1,
            end_var="n",
            statements=(
                make_assignment(
                    target=make_index("arr", make_identifier("i")),
                    value=make_binary(
                        make_index("arr", make_identifier("i")),
                        BinaryOperator.ADD,
                        make_index(
                            "arr",
                            make_binary(
                                make_identifier("i"),
                                BinaryOperator.SUB,
                                make_int(1),
                            ),
                        ),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze_loop(loop)

        assert not analysis.is_parallelizable
        assert len(analysis.dependencies) > 0
