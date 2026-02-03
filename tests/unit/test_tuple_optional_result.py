"""
Unit tests for Tuple, Optional, and Result types in MathViz compiler.
"""

import pytest

from mathviz.compiler.ast_nodes import (
    TupleLiteral,
    SomeExpression,
    OkExpression,
    ErrExpression,
    UnwrapExpression,
    IntegerLiteral,
    StringLiteral,
    Identifier,
)
from mathviz.compiler.type_checker import (
    TupleType,
    OptionalType,
    ResultType,
    INT_TYPE,
    FLOAT_TYPE,
    STRING_TYPE,
    BOOL_TYPE,
    NONE_TYPE,
    EMPTY_TUPLE_TYPE,
    UNKNOWN_TYPE,
)
from mathviz.utils.errors import ParserError


# =============================================================================
# Tuple Tests
# =============================================================================


class TestTupleParsingAndLexing:
    """Tests for tuple literal parsing and lexing."""

    def test_empty_tuple(self, parse):
        """Empty parentheses produce an empty tuple."""
        ast = parse("let x = ()")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 0

    def test_single_element_tuple(self, parse):
        """Single element with trailing comma produces a tuple."""
        ast = parse("let x = (1,)")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 1
        assert isinstance(stmt.value.elements[0], IntegerLiteral)
        assert stmt.value.elements[0].value == 1

    def test_grouped_expression_not_tuple(self, parse):
        """Single element without trailing comma is grouped expression."""
        ast = parse("let x = (1)")
        stmt = ast.statements[0]
        # Should be unwrapped to just the integer, not a tuple
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 1

    def test_two_element_tuple(self, parse):
        """Two elements produce a tuple."""
        ast = parse("let x = (1, 2)")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 2
        assert stmt.value.elements[0].value == 1
        assert stmt.value.elements[1].value == 2

    def test_multi_element_tuple(self, parse):
        """Multiple elements produce a tuple."""
        ast = parse('let x = (1, "hello", 3.14)')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 3

    def test_nested_tuple(self, parse):
        """Nested tuples are supported."""
        ast = parse("let x = ((1, 2), (3, 4))")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 2
        assert isinstance(stmt.value.elements[0], TupleLiteral)
        assert isinstance(stmt.value.elements[1], TupleLiteral)

    def test_tuple_with_trailing_comma(self, parse):
        """Trailing comma is allowed in tuples."""
        ast = parse("let x = (1, 2, 3,)")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 3


class TestTupleTypeChecking:
    """Tests for tuple type checking."""

    def test_empty_tuple_type(self):
        """Empty tuple has special empty tuple type."""
        assert EMPTY_TUPLE_TYPE == TupleType(())
        assert str(EMPTY_TUPLE_TYPE) == "()"

    def test_single_element_tuple_type(self):
        """Single element tuple type has trailing comma in string repr."""
        t = TupleType((INT_TYPE,))
        assert str(t) == "(Int,)"

    def test_multi_element_tuple_type(self):
        """Multi-element tuple type displays properly."""
        t = TupleType((INT_TYPE, STRING_TYPE, FLOAT_TYPE))
        assert str(t) == "(Int, String, Float)"

    def test_tuple_type_compatibility(self):
        """Tuple types are compatible if element types match."""
        t1 = TupleType((INT_TYPE, STRING_TYPE))
        t2 = TupleType((INT_TYPE, STRING_TYPE))
        t3 = TupleType((STRING_TYPE, INT_TYPE))
        t4 = TupleType((INT_TYPE,))

        assert t1.is_compatible_with(t2)
        assert not t1.is_compatible_with(t3)  # Different order
        assert not t1.is_compatible_with(t4)  # Different length

    def test_tuple_type_inference(self, compile_with_analysis):
        """Type checker infers correct tuple types."""
        result = compile_with_analysis('let x = (1, "hello", true)')
        assert not result.has_type_errors()


class TestTupleCodeGeneration:
    """Tests for tuple code generation."""

    def test_empty_tuple_codegen(self, compile_source):
        """Empty tuple generates correctly."""
        code = compile_source("let x = ()")
        assert "x = ()" in code

    def test_single_element_tuple_codegen(self, compile_source):
        """Single element tuple has trailing comma."""
        code = compile_source("let x = (1,)")
        assert "x = (1,)" in code

    def test_multi_element_tuple_codegen(self, compile_source):
        """Multi-element tuple generates correctly."""
        code = compile_source("let x = (1, 2, 3)")
        assert "x = (1, 2, 3)" in code


# =============================================================================
# Optional Tests
# =============================================================================


class TestOptionalParsing:
    """Tests for Optional type parsing."""

    def test_some_expression(self, parse):
        """Some(value) is parsed correctly."""
        ast = parse("let x = Some(42)")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, SomeExpression)
        assert isinstance(stmt.value.value, IntegerLiteral)
        assert stmt.value.value.value == 42

    def test_some_with_string(self, parse):
        """Some with string value."""
        ast = parse('let x = Some("hello")')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, SomeExpression)
        assert isinstance(stmt.value.value, StringLiteral)

    def test_optional_type_annotation(self, parse):
        """Optional[T] type annotation is parsed."""
        ast = parse("let x: Optional[Int] = Some(42)")
        stmt = ast.statements[0]
        assert stmt.type_annotation.base == "Optional"
        assert len(stmt.type_annotation.type_args) == 1


class TestOptionalTypeChecking:
    """Tests for Optional type checking."""

    def test_optional_type_creation(self):
        """OptionalType is created correctly."""
        opt = OptionalType(INT_TYPE)
        assert str(opt) == "Optional[Int]"
        assert opt.inner_type == INT_TYPE

    def test_optional_unwrap_type(self):
        """unwrap_type returns the inner type."""
        opt = OptionalType(STRING_TYPE)
        assert opt.unwrap_type() == STRING_TYPE

    def test_optional_compatible_with_none(self):
        """Optional[T] is compatible with None."""
        opt = OptionalType(INT_TYPE)
        assert opt.is_compatible_with(NONE_TYPE)

    def test_optional_compatible_with_inner_type(self):
        """Optional[T] is compatible with T."""
        opt = OptionalType(INT_TYPE)
        assert opt.is_compatible_with(INT_TYPE)

    def test_optional_compatible_with_same_optional(self):
        """Optional[T] is compatible with Optional[T]."""
        opt1 = OptionalType(INT_TYPE)
        opt2 = OptionalType(INT_TYPE)
        assert opt1.is_compatible_with(opt2)

    def test_optional_not_compatible_with_different_optional(self):
        """Optional[T] is not compatible with Optional[U]."""
        opt1 = OptionalType(INT_TYPE)
        opt2 = OptionalType(STRING_TYPE)
        assert not opt1.is_compatible_with(opt2)


class TestOptionalCodeGeneration:
    """Tests for Optional code generation."""

    def test_some_generates_value(self, compile_source):
        """Some(value) generates just the value in Python."""
        code = compile_source("let x = Some(42)")
        assert "x = 42" in code

    def test_some_with_complex_expression(self, compile_source):
        """Some with complex expression."""
        code = compile_source("let x = Some(1 + 2)")
        assert "x = (1 + 2)" in code


# =============================================================================
# Result Tests
# =============================================================================


class TestResultParsing:
    """Tests for Result type parsing."""

    def test_ok_expression(self, parse):
        """Ok(value) is parsed correctly."""
        ast = parse("let x = Ok(42)")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, OkExpression)
        assert isinstance(stmt.value.value, IntegerLiteral)
        assert stmt.value.value.value == 42

    def test_err_expression(self, parse):
        """Err(error) is parsed correctly."""
        ast = parse('let x = Err("something went wrong")')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ErrExpression)
        assert isinstance(stmt.value.value, StringLiteral)

    def test_result_type_annotation(self, parse):
        """Result[T, E] type annotation is parsed."""
        ast = parse("let x: Result[Int, String] = Ok(42)")
        stmt = ast.statements[0]
        assert stmt.type_annotation.base == "Result"
        assert len(stmt.type_annotation.type_args) == 2


class TestResultTypeChecking:
    """Tests for Result type checking."""

    def test_result_type_creation(self):
        """ResultType is created correctly."""
        result = ResultType(INT_TYPE, STRING_TYPE)
        assert str(result) == "Result[Int, String]"
        assert result.ok_type == INT_TYPE
        assert result.err_type == STRING_TYPE

    def test_result_unwrap_type(self):
        """unwrap_type returns the Ok type."""
        result = ResultType(INT_TYPE, STRING_TYPE)
        assert result.unwrap_type() == INT_TYPE

    def test_result_error_type(self):
        """error_type returns the Err type."""
        result = ResultType(INT_TYPE, STRING_TYPE)
        assert result.error_type() == STRING_TYPE

    def test_result_compatible_with_same_result(self):
        """Result[T, E] is compatible with Result[T, E]."""
        r1 = ResultType(INT_TYPE, STRING_TYPE)
        r2 = ResultType(INT_TYPE, STRING_TYPE)
        assert r1.is_compatible_with(r2)

    def test_result_not_compatible_with_different_ok_type(self):
        """Result[T, E] is not compatible with Result[U, E] when U is incompatible with T.

        Note: Result[Int, E] IS compatible with Result[Float, E] due to Int->Float
        implicit conversion. We test with truly incompatible types instead.
        """
        r1 = ResultType(STRING_TYPE, STRING_TYPE)  # Result[String, String]
        r2 = ResultType(INT_TYPE, STRING_TYPE)  # Result[Int, String]
        # String is not compatible with Int
        assert not r1.is_compatible_with(r2)


class TestResultCodeGeneration:
    """Tests for Result code generation."""

    def test_ok_generates_tuple(self, compile_source):
        """Ok(value) generates (True, value) tuple."""
        code = compile_source("let x = Ok(42)")
        assert "x = (True, 42)" in code

    def test_err_generates_tuple(self, compile_source):
        """Err(error) generates (False, error) tuple."""
        code = compile_source('let x = Err("oops")')
        assert 'x = (False, "oops")' in code or "x = (False, 'oops')" in code


# =============================================================================
# Unwrap Tests
# =============================================================================


class TestUnwrapParsing:
    """Tests for unwrap operator (?) parsing."""

    def test_unwrap_identifier(self, parse):
        """Identifier? is parsed as unwrap."""
        ast = parse("let y = x?")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, UnwrapExpression)
        assert isinstance(stmt.value.operand, Identifier)
        assert stmt.value.operand.name == "x"

    def test_chained_unwrap(self, parse):
        """Multiple unwraps can be chained."""
        ast = parse("let z = x??")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, UnwrapExpression)
        assert isinstance(stmt.value.operand, UnwrapExpression)


class TestUnwrapTypeChecking:
    """Tests for unwrap type checking."""

    def test_unwrap_optional_type_inference(self, compile_with_analysis):
        """Unwrapping Optional[T] yields T."""
        source = """
let x: Optional[Int] = Some(42)
let y = x?
"""
        result = compile_with_analysis(source)
        # The type of y should be inferred as Int (unwrapped Optional)
        assert not result.has_type_errors()


class TestUnwrapCodeGeneration:
    """Tests for unwrap code generation."""

    def test_unwrap_generates_helper_call(self, compile_source):
        """Unwrap generates _mviz_unwrap call."""
        code = compile_source("let y = x?")
        assert "_mviz_unwrap(x)" in code

    def test_unwrap_runtime_injected(self, compile_source):
        """Unwrap usage injects runtime helpers."""
        code = compile_source("let y = x?")
        assert "_mviz_unwrap" in code
        assert "def _mviz_unwrap" in code


# =============================================================================
# Integration Tests
# =============================================================================


class TestTupleOptionalResultIntegration:
    """Integration tests combining multiple features."""

    def test_tuple_of_optionals(self, parse):
        """Tuple containing Optional values."""
        ast = parse("let x = (Some(1), Some(2), None)")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, TupleLiteral)
        assert len(stmt.value.elements) == 3

    def test_optional_tuple(self, compile_with_analysis):
        """Optional containing a tuple."""
        source = "let x: Optional[Tuple[Int, Int]] = Some((1, 2))"
        result = compile_with_analysis(source)
        # Should parse and type check without errors
        assert len(result.ast.statements) == 1

    def test_result_with_tuple_ok_type(self, compile_with_analysis):
        """Result with tuple as Ok type."""
        source = "let x: Result[Tuple[Int, Int], String] = Ok((1, 2))"
        result = compile_with_analysis(source)
        assert len(result.ast.statements) == 1

    def test_function_returning_result(self, compile_source):
        """Function returning Result type."""
        source = """
fn divide(a: Int, b: Int) -> Result[Int, String] {
    if b == 0 {
        return Err("division by zero")
    }
    return Ok(a / b)
}
"""
        code = compile_source(source)
        assert "def divide" in code
        assert "(True," in code
        assert "(False," in code

    def test_function_returning_optional(self, compile_source):
        """Function returning Optional type."""
        source = """
fn find_first(items: List[Int], target: Int) -> Optional[Int] {
    for i in items {
        if i == target {
            return Some(i)
        }
    }
    return None
}
"""
        code = compile_source(source)
        assert "def find_first" in code
        assert "return None" in code

    def test_tuple_destructuring_in_for_loop(self, parse):
        """Tuple patterns in for loop (future feature, should parse tuple)."""
        # For now, just test that tuples in expressions work
        ast = parse("""
let pairs = [(1, 2), (3, 4)]
for pair in pairs {
    let x = pair
}
""")
        assert len(ast.statements) == 2
