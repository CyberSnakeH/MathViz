"""
Unit tests for MathViz constant folding and propagation optimizer.

Tests cover:
- Constant folding of expressions
- Constant propagation through variables
- Dead code elimination
- Algebraic simplification
- Strength reduction
- Common subexpression elimination
- Full optimization pipeline
"""

import pytest
import math

from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.ast_nodes import (
    Program,
    LetStatement,
    ExpressionStatement,
    IfStatement,
    FunctionDef,
    ReturnStatement,
    Block,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BooleanLiteral,
    BinaryExpression,
    UnaryExpression,
    BinaryOperator,
    UnaryOperator,
    Identifier,
    CallExpression,
)
from mathviz.compiler.const_fold import (
    ConstantFolder,
    ConstantPropagator,
    DeadCodeEliminator,
    AlgebraicSimplifier,
    StrengthReducer,
    CSEliminator,
    ConstOptimizer,
    ConstantScope,
    ExpressionKey,
    fold_constants,
    propagate_constants,
    eliminate_dead_code,
    simplify_algebra,
    reduce_strength,
    eliminate_cse,
    optimize_program,
)


def parse_program(source: str) -> Program:
    """Helper to parse a MathViz program."""
    tokens = Lexer(source).tokenize()
    parser = Parser(tokens, source=source)
    return parser.parse()


def parse_expression(source: str):
    """Helper to parse an expression."""
    tokens = Lexer(source).tokenize()
    parser = Parser(tokens, source=source)
    return parser._parse_expression()


class TestConstantFolder:
    """Tests for the ConstantFolder class."""

    def test_fold_integer_addition(self):
        """Test folding integer addition."""
        program = parse_program("let x = 2 + 3")
        folder = ConstantFolder()
        result = folder.fold(program)

        assert len(result.statements) == 1
        stmt = result.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 5

    def test_fold_float_multiplication(self):
        """Test folding float multiplication."""
        program = parse_program("let x = 2.5 * 4.0")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, FloatLiteral)
        assert stmt.value.value == 10.0

    def test_fold_string_concatenation(self):
        """Test folding string concatenation."""
        program = parse_program('let x = "hello" + " world"')
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, StringLiteral)
        assert stmt.value.value == "hello world"

    def test_fold_boolean_and(self):
        """Test folding boolean AND."""
        program = parse_program("let x = true and false")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BooleanLiteral)
        assert stmt.value.value is False

    def test_fold_boolean_or(self):
        """Test folding boolean OR."""
        program = parse_program("let x = true or false")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BooleanLiteral)
        assert stmt.value.value is True

    def test_fold_comparison(self):
        """Test folding comparison operators."""
        test_cases = [
            ("let x = 5 > 3", True),
            ("let x = 5 < 3", False),
            ("let x = 5 == 5", True),
            ("let x = 5 != 5", False),
            ("let x = 5 >= 5", True),
            ("let x = 5 <= 4", False),
        ]
        folder = ConstantFolder()
        for source, expected in test_cases:
            program = parse_program(source)
            result = folder.fold(program)
            stmt = result.statements[0]
            assert isinstance(stmt.value, BooleanLiteral), f"Failed for {source}"
            assert stmt.value.value == expected, f"Failed for {source}"

    def test_fold_unary_negation(self):
        """Test folding unary negation."""
        program = parse_program("let x = -5")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == -5

    def test_fold_double_negation(self):
        """Test folding double negation."""
        program = parse_program("let x = -(-5)")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 5

    def test_fold_not_true(self):
        """Test folding not true."""
        program = parse_program("let x = not true")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BooleanLiteral)
        assert stmt.value.value is False

    def test_fold_nested_expression(self):
        """Test folding nested constant expressions."""
        program = parse_program("let x = (2 + 3) * (4 - 1)")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 15

    def test_fold_with_builtin_constant(self):
        """Test folding with built-in constants like PI."""
        program = parse_program("let x = 2.0 * PI")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, FloatLiteral)
        assert abs(stmt.value.value - 2 * math.pi) < 1e-10

    def test_fold_preserves_non_constant(self):
        """Test that non-constant expressions are preserved."""
        program = parse_program("let x = y + 3")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BinaryExpression)

    def test_fold_conditional_true_condition(self):
        """Test folding conditional expression with true condition."""
        # Create a ConditionalExpression programmatically since MathViz doesn't
        # have a syntax for ternary expressions
        from mathviz.compiler.ast_nodes import ConditionalExpression

        expr = ConditionalExpression(
            condition=BooleanLiteral(value=True),
            then_expr=IntegerLiteral(value=1),
            else_expr=IntegerLiteral(value=2),
        )
        program = Program(statements=(LetStatement(name="x", value=expr),))

        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 1

    def test_fold_conditional_false_condition(self):
        """Test folding conditional expression with false condition."""
        from mathviz.compiler.ast_nodes import ConditionalExpression

        expr = ConditionalExpression(
            condition=BooleanLiteral(value=False),
            then_expr=IntegerLiteral(value=1),
            else_expr=IntegerLiteral(value=2),
        )
        program = Program(statements=(LetStatement(name="x", value=expr),))

        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 2

    def test_fold_in_function(self):
        """Test folding constants inside functions."""
        program = parse_program("""
fn foo() -> Int {
    return 2 + 3
}
""")
        folder = ConstantFolder()
        result = folder.fold(program)

        func = result.statements[0]
        assert isinstance(func, FunctionDef)
        ret_stmt = func.body.statements[0]
        assert isinstance(ret_stmt, ReturnStatement)
        assert isinstance(ret_stmt.value, IntegerLiteral)
        assert ret_stmt.value.value == 5


class TestConstantPropagator:
    """Tests for the ConstantPropagator class."""

    def test_propagate_simple_constant(self):
        """Test propagating a simple constant."""
        program = parse_program("""
let x = 5
let y = x + 3
""")
        propagator = ConstantPropagator()
        result = propagator.propagate(program)

        stmt = result.statements[1]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 8

    def test_propagate_const_declaration(self):
        """Test propagating from const declarations."""
        program = parse_program("""
const PI = 3.14159
let circumference = 2.0 * PI
""")
        propagator = ConstantPropagator()
        result = propagator.propagate(program)

        stmt = result.statements[1]
        assert isinstance(stmt.value, FloatLiteral)
        assert abs(stmt.value.value - 2.0 * 3.14159) < 1e-10

    def test_propagate_chain(self):
        """Test propagating through a chain of constants."""
        program = parse_program("""
let a = 2
let b = a * 3
let c = b + 1
""")
        propagator = ConstantPropagator()
        result = propagator.propagate(program)

        stmt_c = result.statements[2]
        assert isinstance(stmt_c.value, IntegerLiteral)
        assert stmt_c.value.value == 7

    def test_propagate_invalidates_on_reassignment(self):
        """Test that reassignment invalidates constant propagation."""
        program = parse_program("""
let x = 5
x = unknown_value
let y = x + 1
""")
        propagator = ConstantPropagator()
        result = propagator.propagate(program)

        # y should not be folded since x was reassigned
        stmt_y = result.statements[2]
        assert isinstance(stmt_y.value, BinaryExpression)

    def test_propagate_scoped_in_function(self):
        """Test that propagation is scoped within functions."""
        program = parse_program("""
const GLOBAL = 10
fn foo(x: Int) -> Int {
    return x + GLOBAL
}
""")
        propagator = ConstantPropagator()
        result = propagator.propagate(program)

        func = result.statements[1]
        ret_stmt = func.body.statements[0]
        # x is a parameter (not constant), but GLOBAL should be propagated
        assert isinstance(ret_stmt.value, BinaryExpression)


class TestDeadCodeEliminator:
    """Tests for the DeadCodeEliminator class."""

    def test_eliminate_after_return(self):
        """Test removing statements after return."""
        program = parse_program("""
fn foo() -> Int {
    return 5
    let x = 10
}
""")
        eliminator = DeadCodeEliminator()
        result = eliminator.eliminate(program)

        func = result.statements[0]
        assert len(func.body.statements) == 1
        assert isinstance(func.body.statements[0], ReturnStatement)

    def test_eliminate_if_true_condition(self):
        """Test simplifying if with constant true condition."""
        program = parse_program("""
fn foo() -> Int {
    if true {
        return 1
    } else {
        return 2
    }
}
""")
        eliminator = DeadCodeEliminator()
        result = eliminator.eliminate(program)

        func = result.statements[0]
        # The if should be replaced with just the then block
        assert len(func.body.statements) == 1
        ret_stmt = func.body.statements[0]
        assert isinstance(ret_stmt, ReturnStatement)
        assert isinstance(ret_stmt.value, IntegerLiteral)
        assert ret_stmt.value.value == 1

    def test_eliminate_if_false_condition(self):
        """Test simplifying if with constant false condition."""
        program = parse_program("""
fn foo() -> Int {
    if false {
        return 1
    } else {
        return 2
    }
}
""")
        eliminator = DeadCodeEliminator()
        result = eliminator.eliminate(program)

        func = result.statements[0]
        assert len(func.body.statements) == 1
        ret_stmt = func.body.statements[0]
        assert isinstance(ret_stmt, ReturnStatement)
        assert ret_stmt.value.value == 2

    def test_eliminate_while_false(self):
        """Test removing while with constant false condition."""
        program = parse_program("""
fn foo() {
    while false {
        println("never")
    }
}
""")
        eliminator = DeadCodeEliminator()
        result = eliminator.eliminate(program)

        func = result.statements[0]
        # While loop should be removed
        assert len(func.body.statements) == 0


class TestAlgebraicSimplifier:
    """Tests for the AlgebraicSimplifier class."""

    def test_simplify_x_plus_zero(self):
        """Test x + 0 -> x."""
        program = parse_program("let y = x + 0")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_zero_plus_x(self):
        """Test 0 + x -> x."""
        program = parse_program("let y = 0 + x")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_x_minus_zero(self):
        """Test x - 0 -> x."""
        program = parse_program("let y = x - 0")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_x_times_zero(self):
        """Test x * 0 -> 0."""
        program = parse_program("let y = x * 0")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 0

    def test_simplify_x_times_one(self):
        """Test x * 1 -> x."""
        program = parse_program("let y = x * 1")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_x_div_one(self):
        """Test x / 1 -> x."""
        program = parse_program("let y = x / 1")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_x_pow_zero(self):
        """Test x ** 0 -> 1."""
        program = parse_program("let y = x ^ 0")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 1

    def test_simplify_x_pow_one(self):
        """Test x ** 1 -> x."""
        program = parse_program("let y = x ^ 1")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_double_negation(self):
        """Test -(-x) -> x."""
        program = parse_program("let y = -(-x)")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_double_not(self):
        """Test not not x -> x."""
        program = parse_program("let y = not not x")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_true_and_x(self):
        """Test True and x -> x."""
        program = parse_program("let y = true and x")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_simplify_false_and_x(self):
        """Test False and x -> False."""
        program = parse_program("let y = false and x")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BooleanLiteral)
        assert stmt.value.value is False

    def test_simplify_true_or_x(self):
        """Test True or x -> True."""
        program = parse_program("let y = true or x")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BooleanLiteral)
        assert stmt.value.value is True

    def test_simplify_false_or_x(self):
        """Test False or x -> x."""
        program = parse_program("let y = false or x")
        simplifier = AlgebraicSimplifier()
        result = simplifier.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"


class TestStrengthReducer:
    """Tests for the StrengthReducer class."""

    def test_reduce_x_times_2(self):
        """Test x * 2 -> x + x."""
        program = parse_program("let y = x * 2")
        reducer = StrengthReducer()
        result = reducer.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BinaryExpression)
        assert stmt.value.operator == BinaryOperator.ADD
        assert isinstance(stmt.value.left, Identifier)
        assert stmt.value.left.name == "x"
        assert isinstance(stmt.value.right, Identifier)
        assert stmt.value.right.name == "x"

    def test_reduce_x_div_2(self):
        """Test x / 2 -> x * 0.5."""
        program = parse_program("let y = x / 2")
        reducer = StrengthReducer()
        result = reducer.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BinaryExpression)
        assert stmt.value.operator == BinaryOperator.MUL
        assert isinstance(stmt.value.right, FloatLiteral)
        assert stmt.value.right.value == 0.5

    def test_reduce_x_pow_2(self):
        """Test x ** 2 -> x * x."""
        program = parse_program("let y = x ^ 2")
        reducer = StrengthReducer()
        result = reducer.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, BinaryExpression)
        assert stmt.value.operator == BinaryOperator.MUL

    def test_reduce_x_pow_half(self):
        """Test x ** 0.5 -> sqrt(x)."""
        program = parse_program("let y = x ^ 0.5")
        reducer = StrengthReducer()
        result = reducer.optimize(program)

        stmt = result.statements[0]
        assert isinstance(stmt.value, CallExpression)
        assert isinstance(stmt.value.callee, Identifier)
        assert stmt.value.callee.name == "sqrt"


class TestCSEliminator:
    """Tests for the CSEliminator class."""

    def test_eliminate_common_subexpression(self):
        """Test eliminating common subexpressions."""
        program = parse_program("""
fn foo(x: Int, y: Int) -> Int {
    let a = x + y
    let b = x + y
    return a + b
}
""")
        eliminator = CSEliminator(min_uses=2)
        result = eliminator.optimize(program)

        func = result.statements[0]
        # Should have a temp variable for x + y
        assert any(
            isinstance(s, LetStatement) and s.name.startswith("_cse_") for s in func.body.statements
        )

    def test_no_cse_for_single_use(self):
        """Test that single-use expressions are not eliminated."""
        program = parse_program("""
fn foo(x: Int, y: Int) -> Int {
    let a = x + y
    return a
}
""")
        eliminator = CSEliminator(min_uses=2)
        result = eliminator.optimize(program)

        func = result.statements[0]
        # Should not have any CSE temp variables
        assert not any(
            isinstance(s, LetStatement) and s.name.startswith("_cse_") for s in func.body.statements
        )


class TestConstantScope:
    """Tests for the ConstantScope helper class."""

    def test_scope_get_set(self):
        """Test basic get/set operations."""
        scope = ConstantScope()
        scope.set("x", 42)
        assert scope.get("x") == 42
        assert scope.has("x")

    def test_scope_parent_lookup(self):
        """Test parent scope lookup."""
        parent = ConstantScope()
        parent.set("x", 42)

        child = parent.child()
        assert child.get("x") == 42
        assert child.has("x")

    def test_scope_shadowing(self):
        """Test variable shadowing in child scope."""
        parent = ConstantScope()
        parent.set("x", 42)

        child = parent.child()
        child.set("x", 100)
        assert child.get("x") == 100
        assert parent.get("x") == 42

    def test_scope_invalidate(self):
        """Test invalidating a constant."""
        scope = ConstantScope()
        scope.set("x", 42)
        scope.invalidate("x")
        assert scope.get("x") is None
        assert not scope.has("x")


class TestExpressionKey:
    """Tests for the ExpressionKey helper class."""

    def test_key_from_literal(self):
        """Test creating key from literal."""
        expr = IntegerLiteral(value=42)
        key = ExpressionKey.from_expression(expr)
        assert key is not None
        assert key.type_name == "Integer"
        assert key.data == (42,)

    def test_key_from_identifier(self):
        """Test creating key from identifier."""
        expr = Identifier(name="x")
        key = ExpressionKey.from_expression(expr)
        assert key is not None
        assert key.type_name == "Identifier"
        assert key.data == ("x",)

    def test_key_from_binary(self):
        """Test creating key from binary expression."""
        expr = BinaryExpression(
            left=Identifier(name="x"),
            operator=BinaryOperator.ADD,
            right=IntegerLiteral(value=1),
        )
        key = ExpressionKey.from_expression(expr)
        assert key is not None
        assert key.type_name == "Binary"

    def test_equal_keys(self):
        """Test that equal expressions produce equal keys."""
        expr1 = BinaryExpression(
            left=Identifier(name="x"),
            operator=BinaryOperator.ADD,
            right=IntegerLiteral(value=1),
        )
        expr2 = BinaryExpression(
            left=Identifier(name="x"),
            operator=BinaryOperator.ADD,
            right=IntegerLiteral(value=1),
        )
        key1 = ExpressionKey.from_expression(expr1)
        key2 = ExpressionKey.from_expression(expr2)
        assert key1 == key2


class TestConstOptimizer:
    """Tests for the main ConstOptimizer class."""

    def test_full_optimization(self):
        """Test full optimization pipeline."""
        program = parse_program("""
const PI = 3.14159
fn area(r: Float) -> Float {
    let x = 2.0 * PI
    return x * r * r
}
""")
        optimizer = ConstOptimizer()
        result = optimizer.optimize(program)

        # Should successfully optimize
        assert len(result.statements) == 2

    def test_optimization_with_dead_code(self):
        """Test optimization removes dead code."""
        program = parse_program("""
fn foo() -> Int {
    if true {
        return 1
    }
    return 2
}
""")
        optimizer = ConstOptimizer()
        result = optimizer.optimize(program)

        func = result.statements[0]
        # Should only have one return statement
        assert len(func.body.statements) == 1

    def test_get_pass_names(self):
        """Test getting pass names."""
        optimizer = ConstOptimizer(
            fold_constants=True,
            propagate_constants=True,
            eliminate_dead_code=False,
            simplify_algebra=True,
            reduce_strength=False,
            eliminate_cse=False,
        )
        names = optimizer.get_pass_names()
        assert "Constant Folding" in names
        assert "Constant Propagation" in names
        assert "Algebraic Simplification" in names
        assert "Dead Code Elimination" not in names

    def test_optimize_expression(self):
        """Test optimizing a single expression."""
        optimizer = ConstOptimizer()
        expr = BinaryExpression(
            left=IntegerLiteral(value=2),
            operator=BinaryOperator.ADD,
            right=IntegerLiteral(value=3),
        )
        result = optimizer.optimize_expression(expr)
        assert isinstance(result, IntegerLiteral)
        assert result.value == 5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_fold_constants_function(self):
        """Test fold_constants convenience function."""
        program = parse_program("let x = 2 + 3")
        result = fold_constants(program)
        stmt = result.statements[0]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 5

    def test_propagate_constants_function(self):
        """Test propagate_constants convenience function."""
        program = parse_program("""
let x = 5
let y = x + 1
""")
        result = propagate_constants(program)
        stmt = result.statements[1]
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 6

    def test_simplify_algebra_function(self):
        """Test simplify_algebra convenience function."""
        program = parse_program("let y = x + 0")
        result = simplify_algebra(program)
        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)

    def test_reduce_strength_function(self):
        """Test reduce_strength convenience function."""
        program = parse_program("let y = x * 2")
        result = reduce_strength(program)
        stmt = result.statements[0]
        assert isinstance(stmt.value, BinaryExpression)
        assert stmt.value.operator == BinaryOperator.ADD

    def test_optimize_program_function(self):
        """Test optimize_program convenience function."""
        # Test that constant folding happens
        program = parse_program("const X = 2 + 3")
        result = optimize_program(program)
        # Should have the const statement with folded value
        assert len(result.statements) >= 1
        from mathviz.compiler.ast_nodes import ConstDeclaration

        stmt = result.statements[0]
        assert isinstance(stmt, ConstDeclaration)
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 5


class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_division_by_zero_not_folded(self):
        """Test that division by zero is not folded."""
        program = parse_program("let x = 5 / 0")
        folder = ConstantFolder()
        result = folder.fold(program)

        stmt = result.statements[0]
        # Should remain as binary expression
        assert isinstance(stmt.value, BinaryExpression)

    def test_empty_program(self):
        """Test optimizing empty program."""
        program = Program(statements=())
        optimizer = ConstOptimizer()
        result = optimizer.optimize(program)
        assert len(result.statements) == 0

    def test_nested_functions(self):
        """Test optimization with nested structures."""
        program = parse_program("""
fn outer() -> Int {
    fn inner() -> Int {
        return 2 + 3
    }
    return inner()
}
""")
        optimizer = ConstOptimizer()
        # Should not crash
        result = optimizer.optimize(program)
        assert len(result.statements) == 1

    def test_multiple_iterations(self):
        """Test that multiple iterations converge."""
        program = parse_program("""
let a = 1
let b = a + 0
let c = b * 1
let d = c + 0
""")
        optimizer = ConstOptimizer(max_iterations=5)
        result = optimizer.optimize(program)

        # All should simplify to 1
        for stmt in result.statements:
            if isinstance(stmt, LetStatement):
                assert isinstance(stmt.value, IntegerLiteral)
                assert stmt.value.value == 1
