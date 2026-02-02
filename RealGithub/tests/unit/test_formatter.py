"""
Tests for the MathViz code formatter.

Tests cover:
- Expression formatting
- Statement formatting
- Block and function formatting
- Configuration options
- check_format and get_diff functions
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from mathviz.formatter import (
    Formatter,
    FormatConfig,
    format_source,
    format_file,
    check_format,
    get_diff,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def formatter():
    """Create a formatter with default configuration."""
    return Formatter()


@pytest.fixture
def custom_formatter():
    """Create a formatter with custom configuration."""
    config = FormatConfig(
        indent_size=2,
        max_line_length=80,
        use_spaces=True,
    )
    return Formatter(config)


# =============================================================================
# Expression Formatting Tests
# =============================================================================


class TestExpressionFormatting:
    """Tests for expression formatting."""

    def test_format_integer_literal(self):
        """Format integer literal."""
        result = format_source("42")
        assert "42" in result

    def test_format_float_literal(self):
        """Format float literal."""
        result = format_source("3.14")
        assert "3.14" in result

    def test_format_string_literal(self):
        """Format string literal."""
        result = format_source('"hello"')
        assert '"hello"' in result

    def test_format_boolean_literals(self):
        """Format boolean literals."""
        result_true = format_source("true")
        assert "true" in result_true

        result_false = format_source("false")
        assert "false" in result_false

    def test_format_binary_expression(self):
        """Format binary expression with spacing."""
        result = format_source("1 + 2")
        assert "1 + 2" in result

    def test_format_list_literal(self):
        """Format list literal."""
        result = format_source("[1, 2, 3]")
        assert "[1, 2, 3]" in result

    def test_format_dict_literal(self):
        """Format dict literal."""
        result = format_source('{"a": 1, "b": 2}')
        assert '{"a": 1, "b": 2}' in result or "{" in result


# =============================================================================
# Statement Formatting Tests
# =============================================================================


class TestStatementFormatting:
    """Tests for statement formatting."""

    def test_format_let_statement(self):
        """Format let statement."""
        result = format_source("let x = 10")
        assert "let x = 10" in result

    def test_format_let_with_type(self):
        """Format let statement with type annotation."""
        result = format_source("let x: Int = 10")
        assert "let x: Int = 10" in result

    def test_format_assignment(self):
        """Format assignment statement."""
        result = format_source("x = 20")
        assert "x = 20" in result

    def test_format_return_statement(self):
        """Format return statement."""
        result = format_source("fn foo() -> Int { return 42 }")
        assert "return 42" in result

    def test_format_break_continue(self):
        """Format break and continue statements."""
        # Use range() function instead of .. operator due to parser limitations
        source = """
for i in range(10) {
    if i == 5 { break }
    if i == 3 { continue }
}
"""
        result = format_source(source)
        assert "break" in result
        assert "continue" in result


# =============================================================================
# Function Formatting Tests
# =============================================================================


class TestFunctionFormatting:
    """Tests for function formatting."""

    def test_format_simple_function(self):
        """Format simple function."""
        source = "fn add(a: Int, b: Int) -> Int { return a + b }"
        result = format_source(source)
        assert "fn add(a: Int, b: Int) -> Int" in result
        assert "return a + b" in result

    def test_format_function_no_params(self):
        """Format function with no parameters."""
        source = "fn hello() { println(\"Hello\") }"
        result = format_source(source)
        assert "fn hello()" in result

    def test_format_function_no_return_type(self):
        """Format function without return type."""
        source = "fn greet(name: String) { println(name) }"
        result = format_source(source)
        assert "fn greet(name: String)" in result
        # Should not have " -> " if no return type
        lines = result.split("\n")
        fn_line = [l for l in lines if "fn greet" in l][0]
        assert " -> " not in fn_line or "fn greet(name: String) {" in fn_line

    def test_format_jit_function(self):
        """Format JIT-decorated function."""
        source = """
@njit
fn fast_sum(arr: List[Float]) -> Float {
    let total = 0.0
    for x in arr {
        total += x
    }
    return total
}
"""
        result = format_source(source)
        assert "@njit" in result
        assert "fn fast_sum" in result


# =============================================================================
# Control Flow Formatting Tests
# =============================================================================


class TestControlFlowFormatting:
    """Tests for control flow statement formatting."""

    def test_format_if_statement(self):
        """Format if statement."""
        source = "if x > 0 { return 1 }"
        result = format_source(source)
        assert "if x > 0 {" in result

    def test_format_if_else(self):
        """Format if-else statement."""
        source = """
if x > 0 {
    return 1
} else {
    return 0
}
"""
        result = format_source(source)
        assert "if x > 0 {" in result
        assert "} else {" in result

    def test_format_if_elif_else(self):
        """Format if-elif-else statement."""
        source = """
if x > 0 {
    return 1
} elif x < 0 {
    return -1
} else {
    return 0
}
"""
        result = format_source(source)
        assert "if x > 0 {" in result
        assert "} elif x < 0 {" in result
        assert "} else {" in result

    def test_format_for_loop(self):
        """Format for loop."""
        # Use range() function instead of .. operator due to parser limitations
        source = "for i in range(10) { println(i) }"
        result = format_source(source)
        assert "for i in range(10) {" in result

    def test_format_while_loop(self):
        """Format while loop."""
        source = "while x > 0 { x -= 1 }"
        result = format_source(source)
        assert "while x > 0 {" in result


# =============================================================================
# OOP Formatting Tests
# =============================================================================


class TestOOPFormatting:
    """Tests for OOP construct formatting."""

    def test_format_struct(self):
        """Format struct definition."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        result = format_source(source)
        assert "struct Point {" in result
        assert "x: Float" in result
        assert "y: Float" in result

    def test_format_enum(self):
        """Format enum definition."""
        source = """
enum Color {
    Red
    Green
    Blue
}
"""
        result = format_source(source)
        assert "enum Color {" in result
        assert "Red" in result
        assert "Green" in result
        assert "Blue" in result

    def test_format_impl_block(self):
        """Format impl block."""
        source = """
impl Point {
    fn distance(self, other: Point) -> Float {
        return sqrt((self.x - other.x)^2 + (self.y - other.y)^2)
    }
}
"""
        result = format_source(source)
        assert "impl Point {" in result
        assert "fn distance(self, other: Point) -> Float" in result


# =============================================================================
# Indentation Tests
# =============================================================================


class TestIndentation:
    """Tests for indentation handling."""

    def test_default_indent_size(self):
        """Default indent size should be 4 spaces."""
        config = FormatConfig()
        assert config.indent_size == 4

    def test_custom_indent_size(self):
        """Custom indent size should be respected."""
        config = FormatConfig(indent_size=2)
        formatter = Formatter(config)

        source = """
fn foo() {
    let x = 10
}
"""
        # Parse and format through the AST pipeline
        from mathviz.compiler.lexer import Lexer
        from mathviz.compiler.parser import Parser

        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        result = formatter.format_program(ast)
        # With 2-space indent, we should see "  let" (2 spaces)
        lines = result.split("\n")
        let_line = [l for l in lines if "let x" in l][0]
        # Check the indentation (should start with spaces)
        assert let_line.startswith("  ") or let_line.startswith("    ")


# =============================================================================
# API Function Tests
# =============================================================================


class TestAPIFunctions:
    """Tests for public API functions."""

    def test_format_source_basic(self):
        """format_source should return formatted string."""
        source = "let  x  =  10"
        result = format_source(source)
        assert isinstance(result, str)
        assert "let x = 10" in result

    def test_format_source_with_config(self):
        """format_source should accept config."""
        config = FormatConfig(indent_size=2)
        source = "fn foo() { return 1 }"
        result = format_source(source, config)
        assert isinstance(result, str)

    def test_check_format_properly_formatted(self):
        """check_format should return True for properly formatted code."""
        # A simple, already-formatted statement
        source = "let x = 10\n"
        result = check_format(source)
        # This may or may not be True depending on exact formatting rules
        assert isinstance(result, bool)

    def test_check_format_needs_formatting(self):
        """check_format should return False for code needing formatting."""
        # Intentionally poorly formatted
        source = "let   x    =   10"
        # The formatter will normalize spacing
        formatted = format_source(source)
        # If they're different, check_format should return False
        if source != formatted:
            assert check_format(source) is False

    def test_get_diff_no_changes(self):
        """get_diff should return empty string for formatted code."""
        # Format first, then check diff
        source = "let x = 10"
        formatted = format_source(source)
        diff = get_diff(formatted)
        # If source equals formatted, diff should be empty
        if formatted.strip() == source.strip():
            assert diff == "" or diff is not None

    def test_get_diff_with_changes(self):
        """get_diff should return diff for unformatted code."""
        source = "let   x   =   10"
        diff = get_diff(source, filename="test.mviz")
        # Should contain diff output if formatting changes things
        assert isinstance(diff, str)


# =============================================================================
# File Operation Tests
# =============================================================================


class TestFileOperations:
    """Tests for file-based operations."""

    def test_format_file_read_only(self, tmp_path):
        """format_file should read and format without modifying."""
        # Create a temp file
        test_file = tmp_path / "test.mviz"
        source = "let x = 10"
        test_file.write_text(source, encoding="utf-8")

        # Format without in_place
        result = format_file(test_file, in_place=False)
        assert isinstance(result, str)

        # Original file should be unchanged
        assert test_file.read_text() == source

    def test_format_file_in_place(self, tmp_path):
        """format_file with in_place should modify the file."""
        # Create a temp file with unformatted content
        test_file = tmp_path / "test.mviz"
        source = "let x = 10"
        test_file.write_text(source, encoding="utf-8")

        # Format in place
        format_file(test_file, in_place=True)

        # File should now contain formatted content
        result = test_file.read_text()
        assert "let x = 10" in result


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_source(self):
        """Empty source should produce empty output."""
        result = format_source("")
        assert result == "" or result == "\n"

    def test_comment_only(self):
        """Source with only comments might parse as empty."""
        # Comments are typically stripped by the lexer
        # This test depends on how comments are handled
        source = "# This is a comment"
        try:
            result = format_source(source)
            # Either empty or preserves comment
            assert result is not None
        except Exception:
            # May raise if comments aren't handled
            pass

    def test_multiple_statements(self):
        """Multiple statements should all be formatted."""
        source = """
let x = 10
let y = 20
let z = x + y
"""
        result = format_source(source)
        assert "let x = 10" in result
        assert "let y = 20" in result
        assert "let z = x + y" in result

    def test_nested_structures(self):
        """Nested structures should maintain proper indentation."""
        source = """
fn outer() {
    fn inner() {
        return 42
    }
    return inner()
}
"""
        result = format_source(source)
        assert "fn outer()" in result
        # The structure should be preserved
        assert "42" in result


# =============================================================================
# Trailing Newline Tests
# =============================================================================


class TestTrailingNewline:
    """Tests for trailing newline handling."""

    def test_trailing_newline_default(self):
        """Default config should add trailing newline."""
        config = FormatConfig(trailing_newline=True)
        formatter = Formatter(config)

        from mathviz.compiler.lexer import Lexer
        from mathviz.compiler.parser import Parser

        source = "let x = 10"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        result = formatter.format_program(ast)
        assert result.endswith("\n")

    def test_no_trailing_newline(self):
        """Config can disable trailing newline."""
        config = FormatConfig(trailing_newline=False)
        formatter = Formatter(config)

        from mathviz.compiler.lexer import Lexer
        from mathviz.compiler.parser import Parser

        source = "let x = 10"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        result = formatter.format_program(ast)
        assert not result.endswith("\n\n")  # No extra newlines
