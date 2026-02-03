"""
Unit tests for MathViz f-string (string interpolation) support.

Tests cover:
- Lexer tokenization of f-strings
- Parser conversion to FString AST nodes
- Code generation of Python f-strings
- Type checking of f-string expressions
"""

import pytest

from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.tokens import TokenType
from mathviz.compiler.ast_nodes import (
    FString,
    FStringLiteral,
    FStringExpression,
    Identifier,
    IntegerLiteral,
    BinaryExpression,
    BinaryOperator,
)
from mathviz.compiler.codegen import CodeGenerator
from mathviz.utils.errors import LexerError


class TestFStringLexer:
    """Tests for f-string lexer tokenization."""

    def test_simple_fstring(self):
        """Test simple f-string with no interpolation."""
        tokens = Lexer('f"hello world"').tokenize()
        assert len(tokens) == 2  # STRING + EOF
        assert tokens[0].type == TokenType.STRING
        # Value is ("fstring", parts) tuple
        assert tokens[0].value[0] == "fstring"
        parts = tokens[0].value[1]
        assert len(parts) == 1
        assert parts[0] == ("literal", "hello world")

    def test_fstring_with_single_expression(self):
        """Test f-string with a single variable interpolation."""
        tokens = Lexer('f"Hello, {name}!"').tokenize()
        assert tokens[0].type == TokenType.STRING
        parts = tokens[0].value[1]
        assert len(parts) == 3
        assert parts[0] == ("literal", "Hello, ")
        assert parts[1] == ("expr", "name")
        assert parts[2] == ("literal", "!")

    def test_fstring_with_multiple_expressions(self):
        """Test f-string with multiple variable interpolations."""
        tokens = Lexer('f"{a} + {b} = {c}"').tokenize()
        parts = tokens[0].value[1]
        assert len(parts) == 5
        assert parts[0] == ("expr", "a")
        assert parts[1] == ("literal", " + ")
        assert parts[2] == ("expr", "b")
        assert parts[3] == ("literal", " = ")
        assert parts[4] == ("expr", "c")

    def test_fstring_with_format_spec(self):
        """Test f-string with format specifier."""
        tokens = Lexer('f"{value:.2f}"').tokenize()
        parts = tokens[0].value[1]
        assert len(parts) == 1
        assert parts[0] == ("expr", "value", ".2f")

    def test_fstring_with_complex_format_spec(self):
        """Test f-string with various format specifiers."""
        test_cases = [
            ('f"{x:05d}"', ("expr", "x", "05d")),
            ('f"{x:x}"', ("expr", "x", "x")),
            ('f"{x:.1%}"', ("expr", "x", ".1%")),
            ('f"{x:>10}"', ("expr", "x", ">10")),
        ]
        for source, expected in test_cases:
            tokens = Lexer(source).tokenize()
            parts = tokens[0].value[1]
            assert parts[0] == expected, f"Failed for {source}"

    def test_fstring_with_expression(self):
        """Test f-string with arithmetic expression."""
        tokens = Lexer('f"Result: {2 + 2}"').tokenize()
        parts = tokens[0].value[1]
        assert len(parts) == 2
        assert parts[0] == ("literal", "Result: ")
        assert parts[1] == ("expr", "2 + 2")

    def test_fstring_escaped_braces(self):
        """Test f-string with escaped braces."""
        tokens = Lexer(r'f"Use \{braces\} like this"').tokenize()
        parts = tokens[0].value[1]
        assert len(parts) == 1
        assert parts[0] == ("literal", "Use {braces} like this")

    def test_fstring_with_single_quotes(self):
        """Test f-string with single quotes."""
        tokens = Lexer("f'Hello, {name}!'").tokenize()
        parts = tokens[0].value[1]
        assert len(parts) == 3
        assert parts[1] == ("expr", "name")

    def test_fstring_escape_sequences(self):
        """Test escape sequences in f-strings."""
        tokens = Lexer(r'f"Line1\nLine2\t{x}"').tokenize()
        parts = tokens[0].value[1]
        assert parts[0] == ("literal", "Line1\nLine2\t")
        assert parts[1] == ("expr", "x")

    def test_unterminated_fstring_error(self):
        """Test that unterminated f-string raises error."""
        with pytest.raises(LexerError, match="Unterminated"):
            Lexer('f"hello').tokenize()

    def test_unterminated_fstring_expression_error(self):
        """Test that unterminated expression in f-string raises error."""
        with pytest.raises(LexerError, match="Unterminated"):
            Lexer('f"hello {name').tokenize()


class TestFStringParser:
    """Tests for f-string parser."""

    def _parse_expr(self, source: str):
        """Helper to parse an expression."""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        return parser._parse_expression()

    def test_parse_simple_fstring(self):
        """Test parsing simple f-string."""
        fstring = self._parse_expr('f"hello"')
        assert isinstance(fstring, FString)
        assert len(fstring.parts) == 1
        assert isinstance(fstring.parts[0], FStringLiteral)
        assert fstring.parts[0].value == "hello"

    def test_parse_fstring_with_variable(self):
        """Test parsing f-string with variable interpolation."""
        fstring = self._parse_expr('f"Hello, {name}!"')
        assert isinstance(fstring, FString)
        assert len(fstring.parts) == 3
        assert isinstance(fstring.parts[0], FStringLiteral)
        assert isinstance(fstring.parts[1], FStringExpression)
        assert isinstance(fstring.parts[2], FStringLiteral)

        expr_part = fstring.parts[1]
        assert isinstance(expr_part.expression, Identifier)
        assert expr_part.expression.name == "name"

    def test_parse_fstring_with_arithmetic(self):
        """Test parsing f-string with arithmetic expression."""
        fstring = self._parse_expr('f"Sum: {a + b}"')
        assert isinstance(fstring, FString)
        assert len(fstring.parts) == 2

        expr_part = fstring.parts[1]
        assert isinstance(expr_part.expression, BinaryExpression)
        assert expr_part.expression.operator == BinaryOperator.ADD

    def test_parse_fstring_with_format_spec(self):
        """Test parsing f-string with format specifier."""
        fstring = self._parse_expr('f"{pi:.4f}"')
        assert isinstance(fstring, FString)
        assert len(fstring.parts) == 1

        expr_part = fstring.parts[0]
        assert isinstance(expr_part, FStringExpression)
        assert expr_part.format_spec == ".4f"


class TestFStringCodegen:
    """Tests for f-string code generation."""

    def _generate(self, source: str) -> str:
        """Helper to generate Python code from expression."""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        expr = parser._parse_expression()
        gen = CodeGenerator()
        return gen._generate_expr(expr)

    def test_generate_simple_fstring(self):
        """Test generating code for simple f-string."""
        code = self._generate('f"hello world"')
        assert code == 'f"hello world"'

    def test_generate_fstring_with_variable(self):
        """Test generating code for f-string with variable."""
        code = self._generate('f"Hello, {name}!"')
        assert code == 'f"Hello, {name}!"'

    def test_generate_fstring_with_format_spec(self):
        """Test generating code for f-string with format specifier."""
        code = self._generate('f"{value:.2f}"')
        assert code == 'f"{value:.2f}"'

    def test_generate_fstring_with_expression(self):
        """Test generating code for f-string with expression."""
        code = self._generate('f"Sum: {x + y}"')
        assert code == 'f"Sum: {(x + y)}"'


class TestFStringIntegration:
    """Integration tests for f-strings in complete programs."""

    def test_fstring_in_let_statement(self):
        """Test f-string assignment in let statement."""
        source = """
let name = "Alice"
let greeting = f"Hello, {name}!"
"""
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        program = parser.parse()

        gen = CodeGenerator(optimize=False)
        code = gen.generate(program)

        # Python code generator may use single or double quotes
        assert "name = " in code and "Alice" in code
        assert 'f"Hello, {name}!"' in code

    def test_fstring_with_builtin_constant(self):
        """Test f-string with built-in constant."""
        source = 'f"Pi is approximately {PI:.4f}"'
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens, source=source)
        expr = parser._parse_expression()

        gen = CodeGenerator()
        code = gen._generate_expr(expr)
        assert "PI" in code
        assert ".4f" in code
