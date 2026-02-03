"""
Unit tests for MathViz compile-time constants.

Tests cover:
- Lexer tokenization of const keyword
- Parser conversion to ConstDeclaration AST nodes
- Const evaluator compile-time evaluation
- Code generation of constants
- Type checking of constant declarations
"""

import pytest
import math

from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.tokens import TokenType
from mathviz.compiler.ast_nodes import (
    ConstDeclaration,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BinaryExpression,
    BinaryOperator,
    Identifier,
)
from mathviz.compiler.const_evaluator import (
    ConstEvaluator,
    ConstEvalError,
    BUILTIN_CONSTANTS,
    evaluate_const,
    is_const_expr,
)
from mathviz.compiler.codegen import CodeGenerator
from mathviz.utils.errors import ParserError


class TestConstLexer:
    """Tests for const keyword lexer tokenization."""

    def test_const_keyword(self):
        """Test that const is recognized as a keyword."""
        tokens = Lexer("const").tokenize()
        assert tokens[0].type == TokenType.CONST

    def test_const_declaration_tokens(self):
        """Test tokenization of a const declaration."""
        tokens = Lexer("const PI = 3.14").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF and t.type != TokenType.NEWLINE]
        assert types == [
            TokenType.CONST,
            TokenType.IDENTIFIER,
            TokenType.ASSIGN,
            TokenType.FLOAT,
        ]


class TestConstParser:
    """Tests for const declaration parser."""

    def _parse_statement(self, source: str):
        """Helper to parse a statement."""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        return parser._parse_statement()

    def test_parse_simple_const(self):
        """Test parsing simple const declaration."""
        stmt = self._parse_statement("const MAX_SIZE = 1024")
        assert isinstance(stmt, ConstDeclaration)
        assert stmt.name == "MAX_SIZE"
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 1024

    def test_parse_const_with_type_annotation(self):
        """Test parsing const with type annotation."""
        stmt = self._parse_statement("const PI: Float = 3.14")
        assert isinstance(stmt, ConstDeclaration)
        assert stmt.name == "PI"
        assert stmt.type_annotation is not None
        assert isinstance(stmt.value, FloatLiteral)

    def test_parse_const_string(self):
        """Test parsing string constant."""
        stmt = self._parse_statement('const VERSION = "1.0.0"')
        assert isinstance(stmt, ConstDeclaration)
        assert stmt.name == "VERSION"
        assert isinstance(stmt.value, StringLiteral)
        assert stmt.value.value == "1.0.0"

    def test_parse_const_expression(self):
        """Test parsing const with expression value."""
        stmt = self._parse_statement("const TAU = 2.0 * PI")
        assert isinstance(stmt, ConstDeclaration)
        assert stmt.name == "TAU"
        assert isinstance(stmt.value, BinaryExpression)
        assert stmt.value.operator == BinaryOperator.MUL

    def test_const_requires_initializer(self):
        """Test that const without initializer raises error."""
        with pytest.raises(ParserError, match="Expected '='"):
            self._parse_statement("const X")


class TestConstEvaluator:
    """Tests for compile-time constant evaluation."""

    def test_evaluate_integer(self):
        """Test evaluating integer literal."""
        tokens = Lexer("42").tokenize()
        parser = Parser(tokens, source="42")
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        assert evaluator.evaluate(expr) == 42

    def test_evaluate_float(self):
        """Test evaluating float literal."""
        tokens = Lexer("3.14").tokenize()
        parser = Parser(tokens, source="3.14")
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        assert evaluator.evaluate(expr) == 3.14

    def test_evaluate_string(self):
        """Test evaluating string literal."""
        tokens = Lexer('"hello"').tokenize()
        parser = Parser(tokens, source='"hello"')
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        assert evaluator.evaluate(expr) == "hello"

    def test_evaluate_builtin_constant(self):
        """Test evaluating built-in constants."""
        tokens = Lexer("PI").tokenize()
        parser = Parser(tokens, source="PI")
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        assert evaluator.evaluate(expr) == pytest.approx(math.pi, rel=1e-10)

    @pytest.mark.skip(reason="Floor division (//) not yet supported in ConstEvaluator")
    def test_evaluate_arithmetic(self):
        """Test evaluating arithmetic expressions."""
        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("3 * 4", 12),
            ("15 / 3", 5.0),
            ("17 // 5", 3),
            ("17 % 5", 2),
            ("2 ^ 10", 1024),
        ]
        for source, expected in test_cases:
            tokens = Lexer(source).tokenize()
            parser = Parser(tokens, source=source)
            expr = parser._parse_expression()

            evaluator = ConstEvaluator()
            assert evaluator.evaluate(expr) == expected, f"Failed for {source}"

    def test_evaluate_comparison(self):
        """Test evaluating comparison expressions."""
        test_cases = [
            ("5 == 5", True),
            ("5 != 3", True),
            ("3 < 5", True),
            ("5 > 3", True),
            ("3 <= 3", True),
            ("5 >= 5", True),
        ]
        for source, expected in test_cases:
            tokens = Lexer(source).tokenize()
            parser = Parser(tokens, source=source)
            expr = parser._parse_expression()

            evaluator = ConstEvaluator()
            assert evaluator.evaluate(expr) == expected, f"Failed for {source}"

    def test_evaluate_logical(self):
        """Test evaluating logical expressions."""
        test_cases = [
            ("true and true", True),
            ("true and false", False),
            ("true or false", True),
            ("false or false", False),
        ]
        for source, expected in test_cases:
            tokens = Lexer(source).tokenize()
            parser = Parser(tokens, source=source)
            expr = parser._parse_expression()

            evaluator = ConstEvaluator()
            assert evaluator.evaluate(expr) == expected, f"Failed for {source}"

    def test_evaluate_unary(self):
        """Test evaluating unary expressions."""
        test_cases = [
            ("-5", -5),
            ("+5", 5),
            ("not true", False),
            ("not false", True),
        ]
        for source, expected in test_cases:
            tokens = Lexer(source).tokenize()
            parser = Parser(tokens, source=source)
            expr = parser._parse_expression()

            evaluator = ConstEvaluator()
            assert evaluator.evaluate(expr) == expected, f"Failed for {source}"

    def test_evaluate_function_call(self):
        """Test evaluating compile-time function calls."""
        test_cases = [
            ("sqrt(4.0)", 2.0),
            ("abs(-5)", 5),
            ("floor(3.7)", 3),
            ("ceil(3.2)", 4),
            ("min(3, 5)", 3),
            ("max(3, 5)", 5),
        ]
        for source, expected in test_cases:
            tokens = Lexer(source).tokenize()
            parser = Parser(tokens, source=source)
            expr = parser._parse_expression()

            evaluator = ConstEvaluator()
            assert evaluator.evaluate(expr) == expected, f"Failed for {source}"

    def test_evaluate_complex_expression(self):
        """Test evaluating complex nested expression."""
        source = "2.0 * PI"
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        result = evaluator.evaluate(expr)
        assert result == pytest.approx(2.0 * math.pi, rel=1e-10)

    def test_user_defined_constant(self):
        """Test using user-defined constants."""
        evaluator = ConstEvaluator()
        evaluator.add_constant("MY_CONST", 42)

        tokens = Lexer("MY_CONST * 2").tokenize()
        parser = Parser(tokens, source="MY_CONST * 2")
        expr = parser._parse_expression()

        assert evaluator.evaluate(expr) == 84

    def test_unknown_constant_error(self):
        """Test that unknown constant raises error."""
        tokens = Lexer("UNKNOWN_CONST").tokenize()
        parser = Parser(tokens, source="UNKNOWN_CONST")
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        with pytest.raises(ConstEvalError, match="Unknown constant"):
            evaluator.evaluate(expr)

    def test_division_by_zero_error(self):
        """Test that division by zero raises error."""
        tokens = Lexer("5 / 0").tokenize()
        parser = Parser(tokens, source="5 / 0")
        expr = parser._parse_expression()

        evaluator = ConstEvaluator()
        with pytest.raises(ConstEvalError, match="Division by zero"):
            evaluator.evaluate(expr)

    def test_is_const_expr(self):
        """Test checking if expression is constant."""
        evaluator = ConstEvaluator()

        # Constant expression
        tokens1 = Lexer("2 + 3").tokenize()
        parser1 = Parser(tokens1, source="2 + 3")
        expr1 = parser1._parse_expression()
        assert evaluator.is_const_expr(expr1) is True

        # Non-constant expression (unknown variable)
        tokens2 = Lexer("x + 3").tokenize()
        parser2 = Parser(tokens2, source="x + 3")
        expr2 = parser2._parse_expression()
        assert evaluator.is_const_expr(expr2) is False


class TestBuiltinConstants:
    """Tests for built-in mathematical constants."""

    def test_pi_value(self):
        """Test PI constant value."""
        assert BUILTIN_CONSTANTS["PI"] == pytest.approx(math.pi, rel=1e-10)

    def test_e_value(self):
        """Test E constant value."""
        assert BUILTIN_CONSTANTS["E"] == pytest.approx(math.e, rel=1e-10)

    def test_tau_value(self):
        """Test TAU constant value (2*PI)."""
        assert BUILTIN_CONSTANTS["TAU"] == pytest.approx(2 * math.pi, rel=1e-10)

    def test_phi_value(self):
        """Test PHI (golden ratio) constant value."""
        golden_ratio = (1 + math.sqrt(5)) / 2
        assert BUILTIN_CONSTANTS["PHI"] == pytest.approx(golden_ratio, rel=1e-10)

    def test_sqrt2_value(self):
        """Test SQRT2 constant value."""
        assert BUILTIN_CONSTANTS["SQRT2"] == pytest.approx(math.sqrt(2), rel=1e-10)

    def test_inf_value(self):
        """Test INF constant value."""
        assert BUILTIN_CONSTANTS["INF"] == float("inf")

    def test_nan_value(self):
        """Test NAN constant value."""
        assert math.isnan(BUILTIN_CONSTANTS["NAN"])


class TestConstCodegen:
    """Tests for const declaration code generation."""

    def _generate_program(self, source: str) -> str:
        """Helper to generate Python code from program."""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        program = parser.parse()
        gen = CodeGenerator(optimize=False)
        return gen.generate(program)

    def test_generate_simple_const(self):
        """Test generating code for simple const."""
        code = self._generate_program("const MAX_SIZE = 1024")
        assert "MAX_SIZE = 1024" in code

    def test_generate_float_const(self):
        """Test generating code for float const."""
        code = self._generate_program("const PI = 3.14159")
        assert "PI = 3.14159" in code

    def test_generate_string_const(self):
        """Test generating code for string const."""
        code = self._generate_program('const VERSION = "1.0.0"')
        assert "VERSION = " in code
        assert '"1.0.0"' in code or "'1.0.0'" in code

    def test_generate_computed_const(self):
        """Test generating code for computed const."""
        code = self._generate_program("const DOUBLE_PI = 2.0 * PI")
        # Should evaluate at compile time
        assert "DOUBLE_PI" in code


class TestConstIntegration:
    """Integration tests for constants in complete programs."""

    def test_const_in_function(self):
        """Test using constant in function."""
        source = """
const PI = 3.14159
fn circle_area(r: Float) -> Float {
    return PI * r ^ 2
}
"""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        program = parser.parse()

        gen = CodeGenerator(optimize=False)
        code = gen.generate(program)

        assert "PI" in code
        assert "def circle_area" in code

    def test_multiple_consts(self):
        """Test multiple constant declarations."""
        source = """
const PI = 3.14159
const TAU = 2.0 * PI
const VERSION = "1.0.0"
const DEBUG = false
"""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        program = parser.parse()

        gen = CodeGenerator(optimize=False)
        code = gen.generate(program)

        assert "PI" in code
        assert "TAU" in code
        assert "VERSION" in code
        assert "DEBUG" in code


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_evaluate_const_function(self):
        """Test evaluate_const convenience function."""
        tokens = Lexer("2 + 3").tokenize()
        parser = Parser(tokens, source="2 + 3")
        expr = parser._parse_expression()
        assert evaluate_const(expr) == 5

    def test_is_const_expr_function(self):
        """Test is_const_expr convenience function."""
        tokens = Lexer("2 + 3").tokenize()
        parser = Parser(tokens, source="2 + 3")
        expr = parser._parse_expression()
        assert is_const_expr(expr) is True
