"""
Unit tests for the MathViz Parser.
"""

import pytest

from mathviz.compiler.ast_nodes import (
    Program,
    LetStatement,
    FunctionDef,
    ClassDef,
    SceneDef,
    IfStatement,
    ForStatement,
    WhileStatement,
    ReturnStatement,
    ImportStatement,
    ExpressionStatement,
    AssignmentStatement,
    PlayStatement,
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    MemberAccess,
    IndexExpression,
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BooleanLiteral,
    ListLiteral,
    SetLiteral,
    DictLiteral,
    BinaryOperator,
    UnaryOperator,
    SimpleType,
    GenericType,
    Block,
)
from mathviz.utils.errors import ParserError


class TestParserBasics:
    """Basic parser functionality tests."""

    def test_empty_program(self, parse):
        """Empty source should produce empty program."""
        ast = parse("")
        assert isinstance(ast, Program)
        assert len(ast.statements) == 0

    def test_multiple_statements(self, parse):
        """Multiple statements should all be parsed."""
        ast = parse("let x = 1\nlet y = 2\nlet z = 3")
        assert len(ast.statements) == 3


class TestParserLetStatements:
    """Tests for let statement parsing."""

    def test_let_with_value(self, parse):
        """Test simple let statement with value."""
        ast = parse("let x = 42")
        assert len(ast.statements) == 1
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert stmt.name == "x"
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 42

    def test_let_with_type(self, parse):
        """Test let statement with type annotation."""
        ast = parse("let x: Int = 42")
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert stmt.name == "x"
        assert isinstance(stmt.type_annotation, SimpleType)
        assert stmt.type_annotation.name == "Int"

    def test_let_with_generic_type(self, parse):
        """Test let statement with generic type."""
        ast = parse("let items: List[Int] = [1, 2, 3]")
        stmt = ast.statements[0]
        assert isinstance(stmt.type_annotation, GenericType)
        assert stmt.type_annotation.base == "List"

    def test_let_without_value(self, parse):
        """Test let statement without initial value."""
        ast = parse("let x: Int")
        stmt = ast.statements[0]
        assert stmt.value is None


class TestParserFunctionDef:
    """Tests for function definition parsing."""

    def test_simple_function(self, parse):
        """Test simple function definition."""
        ast = parse("fn greet() { return 42 }")
        stmt = ast.statements[0]
        assert isinstance(stmt, FunctionDef)
        assert stmt.name == "greet"
        assert len(stmt.parameters) == 0

    def test_function_with_params(self, parse):
        """Test function with parameters."""
        ast = parse("fn add(x: Int, y: Int) { return x }")
        stmt = ast.statements[0]
        assert len(stmt.parameters) == 2
        assert stmt.parameters[0].name == "x"
        assert stmt.parameters[1].name == "y"

    def test_function_with_return_type(self, parse):
        """Test function with return type annotation."""
        ast = parse("fn add(x: Int, y: Int) -> Int { return x }")
        stmt = ast.statements[0]
        assert isinstance(stmt.return_type, SimpleType)
        assert stmt.return_type.name == "Int"

    def test_function_with_default_params(self, parse):
        """Test function with default parameter values."""
        ast = parse("fn greet(name: String = \"World\") { return name }")
        stmt = ast.statements[0]
        assert stmt.parameters[0].default_value is not None


class TestParserClassDef:
    """Tests for class definition parsing."""

    def test_simple_class(self, parse):
        """Test simple class definition."""
        ast = parse("class Point { let x: Float }")
        stmt = ast.statements[0]
        assert isinstance(stmt, ClassDef)
        assert stmt.name == "Point"

    def test_class_with_base(self, parse):
        """Test class with base class."""
        ast = parse("class Circle(Shape) { let radius: Float }")
        stmt = ast.statements[0]
        assert stmt.base_classes == ("Shape",)


class TestParserSceneDef:
    """Tests for scene definition parsing."""

    def test_simple_scene(self, parse):
        """Test simple scene definition."""
        ast = parse("scene MyAnimation { let x = 1 }")
        stmt = ast.statements[0]
        assert isinstance(stmt, SceneDef)
        assert stmt.name == "MyAnimation"


class TestParserControlFlow:
    """Tests for control flow statement parsing."""

    def test_if_statement(self, parse):
        """Test if statement."""
        ast = parse("if x > 0 { let y = 1 }")
        stmt = ast.statements[0]
        assert isinstance(stmt, IfStatement)
        assert isinstance(stmt.condition, BinaryExpression)

    def test_if_else_statement(self, parse):
        """Test if-else statement."""
        ast = parse("if x > 0 { let y = 1 } else { let y = 2 }")
        stmt = ast.statements[0]
        assert stmt.else_block is not None

    def test_if_elif_else(self, parse):
        """Test if-elif-else chain."""
        ast = parse("if x > 0 { pass } elif x < 0 { pass } else { pass }")
        stmt = ast.statements[0]
        assert len(stmt.elif_clauses) == 1
        assert stmt.else_block is not None

    def test_for_loop(self, parse):
        """Test for loop."""
        ast = parse("for i in items { let x = i }")
        stmt = ast.statements[0]
        assert isinstance(stmt, ForStatement)
        assert stmt.variable == "i"

    def test_while_loop(self, parse):
        """Test while loop."""
        ast = parse("while x > 0 { x = x - 1 }")
        stmt = ast.statements[0]
        assert isinstance(stmt, WhileStatement)


class TestParserExpressions:
    """Tests for expression parsing."""

    def test_binary_arithmetic(self, parse):
        """Test binary arithmetic expressions."""
        ast = parse("let x = 1 + 2")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, BinaryExpression)
        assert stmt.value.operator == BinaryOperator.ADD

    def test_operator_precedence(self, parse):
        """Test operator precedence (multiplication before addition)."""
        ast = parse("let x = 1 + 2 * 3")
        stmt = ast.statements[0]
        # Should be parsed as 1 + (2 * 3), so root is ADD
        assert stmt.value.operator == BinaryOperator.ADD
        assert stmt.value.right.operator == BinaryOperator.MUL

    def test_power_right_associativity(self, parse):
        """Test that power operator is right-associative."""
        ast = parse("let x = 2 ^ 3 ^ 4")
        stmt = ast.statements[0]
        # Should be 2 ^ (3 ^ 4)
        assert stmt.value.operator == BinaryOperator.POW
        assert stmt.value.right.operator == BinaryOperator.POW

    def test_unary_minus(self, parse):
        """Test unary minus."""
        ast = parse("let x = -5")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, UnaryExpression)
        assert stmt.value.operator == UnaryOperator.NEG

    def test_logical_not(self, parse):
        """Test logical not."""
        ast = parse("let x = not true")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, UnaryExpression)
        assert stmt.value.operator == UnaryOperator.NOT

    def test_function_call(self, parse):
        """Test function call expression."""
        ast = parse("foo(42)")
        stmt = ast.statements[0]
        assert isinstance(stmt, ExpressionStatement)
        assert isinstance(stmt.expression, CallExpression)
        assert isinstance(stmt.expression.callee, Identifier)
        assert stmt.expression.callee.name == "foo"

    def test_method_call(self, parse):
        """Test method call expression."""
        ast = parse("obj.method(1, 2)")
        stmt = ast.statements[0]
        call = stmt.expression
        assert isinstance(call.callee, MemberAccess)
        assert call.callee.member == "method"

    def test_index_access(self, parse):
        """Test index access expression."""
        ast = parse("let x = arr[0]")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, IndexExpression)

    def test_chained_member_access(self, parse):
        """Test chained member access."""
        ast = parse("let x = a.b.c")
        stmt = ast.statements[0]
        # Should be ((a.b).c)
        outer = stmt.value
        assert isinstance(outer, MemberAccess)
        assert outer.member == "c"
        inner = outer.object
        assert isinstance(inner, MemberAccess)
        assert inner.member == "b"


class TestParserMathOperators:
    """Tests for mathematical operator parsing."""

    def test_element_of(self, parse):
        """Test element-of operator."""
        ast = parse("let x = 1 ∈ S")
        stmt = ast.statements[0]
        assert stmt.value.operator == BinaryOperator.ELEMENT_OF

    def test_not_element_of(self, parse):
        """Test not-element-of operator."""
        ast = parse("let x = 1 ∉ S")
        stmt = ast.statements[0]
        assert stmt.value.operator == BinaryOperator.NOT_ELEMENT_OF

    def test_subset(self, parse):
        """Test subset operator."""
        ast = parse("let x = A ⊆ B")
        stmt = ast.statements[0]
        assert stmt.value.operator == BinaryOperator.SUBSET

    def test_union(self, parse):
        """Test union operator."""
        ast = parse("let x = A ∪ B")
        stmt = ast.statements[0]
        assert stmt.value.operator == BinaryOperator.UNION

    def test_intersection(self, parse):
        """Test intersection operator."""
        ast = parse("let x = A ∩ B")
        stmt = ast.statements[0]
        assert stmt.value.operator == BinaryOperator.INTERSECTION


class TestParserLiterals:
    """Tests for literal parsing."""

    def test_list_literal(self, parse):
        """Test list literal."""
        ast = parse("let x = [1, 2, 3]")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ListLiteral)
        assert len(stmt.value.elements) == 3

    def test_set_literal(self, parse):
        """Test set literal."""
        ast = parse("let x = {1, 2, 3}")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, SetLiteral)
        assert len(stmt.value.elements) == 3

    def test_dict_literal(self, parse):
        """Test dictionary literal."""
        ast = parse('let x = {"a": 1, "b": 2}')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, DictLiteral)
        assert len(stmt.value.pairs) == 2

    def test_empty_dict(self, parse):
        """Test empty dictionary literal."""
        ast = parse("let x = {}")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, DictLiteral)
        assert len(stmt.value.pairs) == 0


class TestParserImports:
    """Tests for import statement parsing."""

    def test_simple_import(self, parse):
        """Test simple import."""
        ast = parse("import manim")
        stmt = ast.statements[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module == "manim"

    def test_import_as(self, parse):
        """Test import with alias."""
        ast = parse("import numpy as np")
        stmt = ast.statements[0]
        assert stmt.module == "numpy"
        assert stmt.alias == "np"

    def test_from_import(self, parse):
        """Test from...import."""
        ast = parse("from manim import Circle, Square")
        stmt = ast.statements[0]
        assert stmt.is_from_import
        assert stmt.module == "manim"
        assert ("Circle", None) in stmt.names
        assert ("Square", None) in stmt.names


class TestParserErrors:
    """Tests for parser error handling."""

    def test_missing_block_open(self, parser_factory):
        """Test error on missing opening brace."""
        parser = parser_factory("if x > 0 pass }")
        with pytest.raises(ParserError):
            parser.parse()

    def test_missing_rparen(self, parser_factory):
        """Test error on missing closing parenthesis."""
        parser = parser_factory("fn foo( { }")
        with pytest.raises(ParserError):
            parser.parse()


class TestParserAnimateProperty:
    """Tests for the animate property (keyword as member name)."""

    def test_simple_animate(self, parse):
        """Test simple animate property access."""
        ast = parse("circle.animate.scale(2)")
        assert len(ast.statements) == 1
        stmt = ast.statements[0]
        assert isinstance(stmt, ExpressionStatement)
        # circle.animate.scale(2) is a call to circle.animate.scale
        call = stmt.expression
        assert isinstance(call, CallExpression)
        # The callee is circle.animate.scale
        assert isinstance(call.callee, MemberAccess)
        assert call.callee.member == "scale"
        # circle.animate
        animate_access = call.callee.object
        assert isinstance(animate_access, MemberAccess)
        assert animate_access.member == "animate"

    def test_chained_animate(self, parse):
        """Test chained animate methods."""
        ast = parse("circle.animate.scale(2).shift(RIGHT)")
        assert len(ast.statements) == 1
        stmt = ast.statements[0]
        assert isinstance(stmt, ExpressionStatement)
        # The outer expression is a call to .shift(RIGHT)
        call = stmt.expression
        assert isinstance(call, CallExpression)
        assert isinstance(call.callee, MemberAccess)
        assert call.callee.member == "shift"

    def test_animate_in_play(self, parse):
        """Test animate inside play statement."""
        ast = parse("play(circle.animate.scale(2))")
        stmt = ast.statements[0]
        assert isinstance(stmt, PlayStatement)

    def test_animate_with_multiple_args(self, parse):
        """Test animate method with multiple arguments."""
        ast = parse("circle.animate.set_color(RED)")
        assert len(ast.statements) == 1
        stmt = ast.statements[0]
        assert isinstance(stmt, ExpressionStatement)
        call = stmt.expression
        assert isinstance(call, CallExpression)
        assert len(call.arguments) == 1
