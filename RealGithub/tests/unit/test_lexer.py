"""
Unit tests for the MathViz Lexer.
"""

import pytest

from mathviz.compiler.lexer import Lexer
from mathviz.compiler.tokens import Token, TokenType
from mathviz.utils.errors import LexerError


class TestLexerBasics:
    """Basic lexer functionality tests."""

    def test_empty_source(self, tokenize):
        """Empty source should produce only EOF token."""
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only(self, tokenize):
        """Whitespace-only source should produce only EOF token."""
        tokens = tokenize("   \t   ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_single_line_comment(self, tokenize):
        """Single-line comments should be skipped."""
        tokens = tokenize("# this is a comment")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_comment_with_code(self, tokenize):
        """Comments should not affect adjacent code."""
        tokens = tokenize("x # comment\ny")
        types = [t.type for t in tokens]
        assert TokenType.IDENTIFIER in types
        assert TokenType.COMMENT not in types


class TestLexerLiterals:
    """Tests for literal tokenization."""

    def test_integer_literals(self, tokenize):
        """Test various integer literal formats."""
        tokens = tokenize("42 0 1_000_000 999")
        integers = [t for t in tokens if t.type == TokenType.INTEGER]
        assert len(integers) == 4
        assert integers[0].value == 42
        assert integers[1].value == 0
        assert integers[2].value == 1000000
        assert integers[3].value == 999

    def test_float_literals(self, tokenize):
        """Test various float literal formats."""
        tokens = tokenize("3.14 0.5 1.0e10 2.5E-3")
        floats = [t for t in tokens if t.type == TokenType.FLOAT]
        assert len(floats) == 4
        assert floats[0].value == 3.14
        assert floats[1].value == 0.5
        assert floats[2].value == 1.0e10
        assert floats[3].value == 2.5e-3

    def test_string_literals_double_quote(self, tokenize):
        """Test double-quoted string literals."""
        tokens = tokenize('"hello world"')
        strings = [t for t in tokens if t.type == TokenType.STRING]
        assert len(strings) == 1
        assert strings[0].value == "hello world"

    def test_string_literals_single_quote(self, tokenize):
        """Test single-quoted string literals."""
        tokens = tokenize("'hello world'")
        strings = [t for t in tokens if t.type == TokenType.STRING]
        assert len(strings) == 1
        assert strings[0].value == "hello world"

    def test_string_escape_sequences(self, tokenize):
        """Test escape sequences in strings."""
        tokens = tokenize(r'"hello\nworld\t!"')
        strings = [t for t in tokens if t.type == TokenType.STRING]
        assert len(strings) == 1
        assert strings[0].value == "hello\nworld\t!"

    def test_boolean_literals(self, tokenize):
        """Test boolean literals."""
        tokens = tokenize("true false True False")
        bools = [t for t in tokens if t.type in (TokenType.TRUE, TokenType.FALSE)]
        assert len(bools) == 4
        assert bools[0].value is True
        assert bools[1].value is False


class TestLexerKeywords:
    """Tests for keyword recognition."""

    def test_all_keywords(self, tokenize):
        """Test that all keywords are recognized."""
        source = "let fn class scene if else elif for while return import from as in"
        tokens = tokenize(source)
        expected = [
            TokenType.LET, TokenType.FN, TokenType.CLASS, TokenType.SCENE,
            TokenType.IF, TokenType.ELSE, TokenType.ELIF, TokenType.FOR,
            TokenType.WHILE, TokenType.RETURN, TokenType.IMPORT, TokenType.FROM,
            TokenType.AS, TokenType.IN,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_type_keywords(self, tokenize):
        """Test type annotation keywords."""
        source = "Int Float Bool String List Set Dict"
        tokens = tokenize(source)
        expected = [
            TokenType.TYPE_INT, TokenType.TYPE_FLOAT, TokenType.TYPE_BOOL,
            TokenType.TYPE_STRING, TokenType.TYPE_LIST, TokenType.TYPE_SET,
            TokenType.TYPE_DICT,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected


class TestLexerOperators:
    """Tests for operator tokenization."""

    def test_arithmetic_operators(self, tokenize):
        """Test arithmetic operators."""
        tokens = tokenize("+ - * / // % ^ **")
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.DOUBLE_SLASH, TokenType.PERCENT, TokenType.CARET,
            TokenType.DOUBLE_STAR,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_comparison_operators(self, tokenize):
        """Test comparison operators."""
        tokens = tokenize("== != < > <= >=")
        expected = [
            TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_assignment_operators(self, tokenize):
        """Test assignment operators."""
        tokens = tokenize("= += -= *= /=")
        expected = [
            TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected


class TestLexerMathSymbols:
    """Tests for mathematical Unicode symbol tokenization."""

    def test_set_membership_operators(self, tokenize):
        """Test set membership operators."""
        tokens = tokenize("∈ ∉")
        expected = [TokenType.ELEMENT_OF, TokenType.NOT_ELEMENT_OF]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_set_relation_operators(self, tokenize):
        """Test set relation operators."""
        tokens = tokenize("⊆ ⊇ ⊂ ⊃")
        expected = [
            TokenType.SUBSET, TokenType.SUPERSET,
            TokenType.PROPER_SUBSET, TokenType.PROPER_SUPERSET,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_set_operation_operators(self, tokenize):
        """Test set operation operators."""
        tokens = tokenize("∪ ∩ ∖")
        expected = [TokenType.UNION, TokenType.INTERSECTION, TokenType.SET_DIFF]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_math_constants(self, tokenize):
        """Test mathematical constants."""
        tokens = tokenize("π ∞ ∅")
        expected = [TokenType.PI, TokenType.INFINITY, TokenType.EMPTY_SET]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_pi_value(self, tokenize):
        """Test that pi has correct value."""
        tokens = tokenize("π")
        pi_token = tokens[0]
        assert pi_token.type == TokenType.PI
        assert abs(pi_token.value - 3.141592653589793) < 1e-10


class TestLexerDelimiters:
    """Tests for delimiter tokenization."""

    def test_parentheses(self, tokenize):
        """Test parentheses."""
        tokens = tokenize("()")
        expected = [TokenType.LPAREN, TokenType.RPAREN]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_braces(self, tokenize):
        """Test curly braces."""
        tokens = tokenize("{}")
        expected = [TokenType.LBRACE, TokenType.RBRACE]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_brackets(self, tokenize):
        """Test square brackets."""
        tokens = tokenize("[]")
        expected = [TokenType.LBRACKET, TokenType.RBRACKET]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected

    def test_punctuation(self, tokenize):
        """Test punctuation marks."""
        tokens = tokenize(", . : ; -> =>")
        expected = [
            TokenType.COMMA, TokenType.DOT, TokenType.COLON,
            TokenType.SEMICOLON, TokenType.THIN_ARROW, TokenType.FAT_ARROW,
        ]
        actual = [t.type for t in tokens if t.type != TokenType.EOF]
        assert actual == expected


class TestLexerErrors:
    """Tests for lexer error handling."""

    def test_unterminated_string(self, lexer_factory):
        """Test error on unterminated string."""
        lexer = lexer_factory('"hello')
        with pytest.raises(LexerError, match="Unterminated string"):
            lexer.tokenize()

    def test_invalid_escape_sequence(self, lexer_factory):
        """Test error on invalid escape sequence."""
        lexer = lexer_factory('"hello\\z"')
        with pytest.raises(LexerError, match="Invalid escape sequence"):
            lexer.tokenize()

    def test_invalid_number_exponent(self, lexer_factory):
        """Test error on invalid number with missing exponent digits."""
        lexer = lexer_factory("1.5e")
        with pytest.raises(LexerError, match="expected exponent digits"):
            lexer.tokenize()


class TestLexerSourceLocations:
    """Tests for source location tracking."""

    def test_location_tracking(self, tokenize):
        """Test that token locations are tracked correctly."""
        tokens = tokenize("x\ny\nz")
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert identifiers[0].location.line == 1
        assert identifiers[1].location.line == 2
        assert identifiers[2].location.line == 3

    def test_column_tracking(self, tokenize):
        """Test column tracking."""
        tokens = tokenize("abc def")
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert identifiers[0].location.column == 1
        assert identifiers[1].location.column == 5
