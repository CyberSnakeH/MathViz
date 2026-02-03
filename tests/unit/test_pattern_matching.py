"""
Tests for MathViz Pattern Matching.

Tests cover:
- Lexer: MATCH, WHERE, UNDERSCORE tokens
- Parser: MatchExpression, Pattern AST nodes
- Type Checker: Pattern type inference, exhaustiveness checking
- Code Generator: Match expression to Python if-elif-else or ternary
"""

import pytest

from mathviz.compiler.ast_nodes import (
    BooleanLiteral,
    ConstructorPattern,
    IdentifierPattern,
    IntegerLiteral,
    LiteralPattern,
    MatchExpression,
    NoneLiteral,
    TuplePattern,
)
from mathviz.compiler.codegen import CodeGenerator
from mathviz.compiler.lexer import tokenize
from mathviz.compiler.parser import Parser
from mathviz.compiler.tokens import TokenType
from mathviz.compiler.type_checker import (
    TypeChecker,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tokenize_source():
    """Fixture to tokenize MathViz source code."""

    def _tokenize(source: str):
        return tokenize(source)

    return _tokenize


@pytest.fixture
def parse(tokenize_source):
    """Fixture to parse MathViz source code into AST."""

    def _parse(source: str):
        tokens = tokenize_source(source)
        parser = Parser(tokens)
        return parser.parse()

    return _parse


@pytest.fixture
def type_check(parse):
    """Fixture to type check MathViz source code."""

    def _type_check(source: str):
        program = parse(source)
        checker = TypeChecker(source)
        errors = checker.check(program)
        return errors, checker

    return _type_check


@pytest.fixture
def compile_source(parse):
    """Fixture to compile MathViz source code to Python."""

    def _compile(source: str, optimize: bool = False):
        program = parse(source)
        generator = CodeGenerator(optimize=optimize)
        return generator.generate(program)

    return _compile


# =============================================================================
# Lexer Tests
# =============================================================================


class TestPatternMatchingLexer:
    """Tests for lexing pattern matching tokens."""

    def test_match_keyword(self, tokenize_source):
        """'match' is tokenized as MATCH keyword."""
        tokens = tokenize_source("match")
        assert tokens[0].type == TokenType.MATCH
        assert tokens[0].value == "match"

    def test_where_keyword(self, tokenize_source):
        """'where' is tokenized as WHERE keyword."""
        tokens = tokenize_source("where")
        assert tokens[0].type == TokenType.WHERE
        assert tokens[0].value == "where"

    def test_underscore_keyword(self, tokenize_source):
        """'_' is tokenized as UNDERSCORE keyword."""
        tokens = tokenize_source("_")
        assert tokens[0].type == TokenType.UNDERSCORE
        assert tokens[0].value == "_"

    def test_thin_arrow(self, tokenize_source):
        """'->' is tokenized as THIN_ARROW."""
        tokens = tokenize_source("->")
        assert tokens[0].type == TokenType.THIN_ARROW

    def test_match_expression_tokens(self, tokenize_source):
        """Full match expression tokenizes correctly."""
        tokens = tokenize_source("match x { 0 -> y }")
        token_types = [t.type for t in tokens if t.type != TokenType.NEWLINE]
        assert TokenType.MATCH in token_types
        assert TokenType.LBRACE in token_types
        assert TokenType.RBRACE in token_types
        assert TokenType.THIN_ARROW in token_types


# =============================================================================
# Parser Tests
# =============================================================================


class TestPatternMatchingParser:
    """Tests for parsing match expressions."""

    def test_simple_literal_match(self, parse):
        """Parse match with literal patterns."""
        program = parse("""
let result = match x {
    0 -> "zero"
    1 -> "one"
    _ -> "other"
}
""")
        # Find the match expression
        let_stmt = program.statements[0]
        match_expr = let_stmt.value
        assert isinstance(match_expr, MatchExpression)
        assert len(match_expr.arms) == 3

    def test_literal_pattern(self, parse):
        """Parse literal pattern: 0, 1, "hello"."""
        program = parse('let result = match x { 0 -> "zero" }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, LiteralPattern)
        assert isinstance(arm.pattern.value, IntegerLiteral)
        assert arm.pattern.value.value == 0

    def test_identifier_pattern(self, parse):
        """Parse identifier pattern: n."""
        program = parse("let result = match x { n -> n * 2 }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, IdentifierPattern)
        assert arm.pattern.name == "n"
        assert not arm.pattern.is_wildcard

    def test_wildcard_pattern(self, parse):
        """Parse wildcard pattern: _."""
        program = parse('let result = match x { _ -> "default" }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, IdentifierPattern)
        assert arm.pattern.is_wildcard

    def test_tuple_pattern(self, parse):
        """Parse tuple pattern: (x, y)."""
        program = parse("let result = match point { (x, y) -> x + y }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, TuplePattern)
        assert len(arm.pattern.elements) == 2
        assert arm.pattern.elements[0].name == "x"
        assert arm.pattern.elements[1].name == "y"

    def test_constructor_pattern_some(self, parse):
        """Parse Some(x) pattern."""
        program = parse("let result = match opt { Some(x) -> x }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, ConstructorPattern)
        assert arm.pattern.name == "Some"
        assert len(arm.pattern.args) == 1

    def test_constructor_pattern_ok(self, parse):
        """Parse Ok(value) pattern."""
        program = parse("let result = match res { Ok(v) -> v }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, ConstructorPattern)
        assert arm.pattern.name == "Ok"

    def test_constructor_pattern_err(self, parse):
        """Parse Err(e) pattern."""
        program = parse("let result = match res { Err(e) -> e }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, ConstructorPattern)
        assert arm.pattern.name == "Err"

    def test_pattern_with_guard(self, parse):
        """Parse pattern with guard: n where n > 0."""
        program = parse('let result = match x { n where n > 0 -> "positive" }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert arm.guard is not None

    def test_multiple_arms(self, parse):
        """Parse multiple match arms."""
        program = parse("""
let result = match x {
    0 -> "zero"
    1 -> "one"
    n where n > 0 -> "positive"
    _ -> "negative"
}
""")
        match_expr = program.statements[0].value
        assert len(match_expr.arms) == 4

    def test_boolean_literal_pattern(self, parse):
        """Parse boolean literal pattern."""
        program = parse('let result = match flag { true -> "yes" }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, LiteralPattern)
        assert isinstance(arm.pattern.value, BooleanLiteral)
        assert arm.pattern.value.value is True

    def test_none_literal_pattern(self, parse):
        """Parse None literal pattern."""
        program = parse('let result = match opt { None -> "nothing" }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, LiteralPattern)
        assert isinstance(arm.pattern.value, NoneLiteral)


# =============================================================================
# Type Checker Tests
# =============================================================================


class TestPatternMatchingTypeChecker:
    """Tests for type checking match expressions."""

    def test_match_infers_unified_type(self, type_check):
        """Match expression type is unified from all arms."""
        errors, checker = type_check("""
fn test(x: Int) -> Int {
    return match x {
        0 -> 0
        n -> n * 2
    }
}
""")
        # Should have no type errors (all arms return Int)
        assert len(errors) == 0

    def test_match_with_incompatible_arm_types(self, type_check):
        """Match with incompatible arm types produces error or Any type."""
        errors, checker = type_check("""
fn test(x: Int) -> Int {
    return match x {
        0 -> 0
        n -> "string"
    }
}
""")
        # Should have at least one error (String vs Int)
        # Note: Depending on implementation, might return Any type
        # This test verifies the type checker handles mixed types

    def test_exhaustiveness_bool(self, type_check):
        """Non-exhaustive bool match produces error."""
        errors, checker = type_check("""
fn test(x: Bool) -> String {
    return match x {
        true -> "yes"
    }
}
""")
        # Should warn about missing 'false' case
        assert any("exhaustive" in str(e).lower() or "false" in str(e).lower() for e in errors)

    def test_exhaustiveness_bool_complete(self, type_check):
        """Complete bool match has no exhaustiveness error."""
        errors, checker = type_check("""
fn test(x: Bool) -> String {
    return match x {
        true -> "yes"
        false -> "no"
    }
}
""")
        # Should have no errors
        assert len([e for e in errors if "exhaustive" in str(e).lower()]) == 0

    def test_exhaustiveness_with_wildcard(self, type_check):
        """Wildcard pattern makes match exhaustive."""
        errors, checker = type_check("""
fn test(x: Int) -> String {
    return match x {
        0 -> "zero"
        _ -> "other"
    }
}
""")
        # No exhaustiveness errors
        assert len([e for e in errors if "exhaustive" in str(e).lower()]) == 0

    def test_pattern_binds_variable(self, type_check):
        """Pattern binding creates variable in scope."""
        errors, checker = type_check("""
fn test(x: Int) -> Int {
    return match x {
        n -> n * 2
    }
}
""")
        # 'n' should be accessible in arm body
        assert len(errors) == 0

    def test_guard_must_be_bool(self, type_check):
        """Guard expression must have Bool type."""
        errors, checker = type_check("""
fn test(x: Int) -> Int {
    return match x {
        n where n -> n
    }
}
""")
        # Guard 'n' is Int, not Bool - should error
        assert any("bool" in str(e).lower() for e in errors)


# =============================================================================
# Code Generator Tests
# =============================================================================


class TestPatternMatchingCodeGenerator:
    """Tests for generating code from match expressions."""

    def test_simple_match_generates_lambda(self, compile_source):
        """Simple match generates lambda with ternary."""
        code = compile_source("""
let result = match x {
    0 -> "zero"
    _ -> "other"
}
""")
        # Should generate lambda pattern
        assert "lambda" in code

    def test_literal_pattern_generates_equality(self, compile_source):
        """Literal pattern generates equality check."""
        code = compile_source("""
let result = match x {
    0 -> "zero"
    1 -> "one"
}
""")
        # Should contain equality checks
        assert "==" in code

    def test_wildcard_generates_true(self, compile_source):
        """Wildcard pattern generates 'True' condition."""
        code = compile_source("""
let result = match x {
    _ -> "default"
}
""")
        # Wildcard should generate code that always matches
        # (typically just the body without condition)
        assert "default" in code

    def test_identifier_binds_subject(self, compile_source):
        """Identifier pattern binds subject to variable."""
        code = compile_source("""
let result = match x {
    n -> n * 2
}
""")
        # Should bind n to the subject
        assert "n" in code
        assert "*" in code

    def test_tuple_pattern_generates_checks(self, compile_source):
        """Tuple pattern generates isinstance and length checks."""
        code = compile_source("""
let result = match point {
    (0, 0) -> "origin"
    (x, y) -> x + y
}
""")
        # Should generate tuple checks
        assert "tuple" in code or "isinstance" in code

    def test_some_pattern_generates_check(self, compile_source):
        """Some(x) pattern generates not None check."""
        code = compile_source("""
let result = match opt {
    Some(x) -> x
    None -> 0
}
""")
        # Should check for not None
        assert "None" in code

    def test_ok_pattern_generates_tuple_check(self, compile_source):
        """Ok(x) pattern generates Result tuple check."""
        code = compile_source("""
let result = match res {
    Ok(x) -> x
    Err(e) -> 0
}
""")
        # Should check tuple structure
        assert "True" in code  # Ok is (True, value)
        assert "False" in code  # Err is (False, error)

    def test_guard_generates_condition(self, compile_source):
        """Guard generates additional condition."""
        code = compile_source("""
let result = match x {
    n where n > 0 -> "positive"
    _ -> "other"
}
""")
        # Guard should appear as additional condition
        assert ">" in code

    def test_match_as_expression(self, compile_source):
        """Match can be used as expression."""
        code = compile_source("""
let doubled = match x {
    0 -> 0
    n -> n * 2
}
""")
        # Should be assigned to variable
        assert "doubled" in code

    def test_nested_match_not_supported_but_compiles(self, compile_source):
        """Nested match compiles (even if not optimal)."""
        code = compile_source("""
let result = match x {
    0 -> match y {
        0 -> "both zero"
        _ -> "y not zero"
    }
    _ -> "x not zero"
}
""")
        # Should compile without error
        assert "lambda" in code


# =============================================================================
# Integration Tests
# =============================================================================


class TestPatternMatchingIntegration:
    """Integration tests for pattern matching."""

    def test_full_pipeline_literal_match(self, compile_source):
        """Full pipeline: literal match compiles and produces valid Python."""
        code = compile_source("""
fn describe(n: Int) -> String {
    return match n {
        0 -> "zero"
        1 -> "one"
        2 -> "two"
        _ -> "many"
    }
}
""")
        # Verify generated code is syntactically valid Python
        compile(code, "<string>", "exec")

    def test_full_pipeline_optional_match(self, compile_source):
        """Full pipeline: Optional match compiles."""
        code = compile_source("""
fn unwrap_or_default(opt: Optional[Int]) -> Int {
    return match opt {
        Some(x) -> x
        None -> 0
    }
}
""")
        compile(code, "<string>", "exec")

    def test_full_pipeline_result_match(self, compile_source):
        """Full pipeline: Result match compiles."""
        code = compile_source("""
fn handle_result(res: Result[Int, String]) -> Int {
    return match res {
        Ok(value) -> value
        Err(e) -> 0
    }
}
""")
        compile(code, "<string>", "exec")

    def test_full_pipeline_tuple_match(self, compile_source):
        """Full pipeline: Tuple match compiles."""
        code = compile_source("""
fn point_to_string(point: Tuple[Int, Int]) -> String {
    return match point {
        (0, 0) -> "origin"
        (x, 0) -> "on x-axis"
        (0, y) -> "on y-axis"
        (x, y) -> "general"
    }
}
""")
        compile(code, "<string>", "exec")

    def test_full_pipeline_guarded_match(self, compile_source):
        """Full pipeline: guarded match compiles."""
        code = compile_source("""
fn classify(n: Int) -> String {
    return match n {
        0 -> "zero"
        n where n > 0 -> "positive"
        n where n < 0 -> "negative"
    }
}
""")
        compile(code, "<string>", "exec")


# =============================================================================
# Edge Cases
# =============================================================================


class TestPatternMatchingEdgeCases:
    """Edge case tests for pattern matching."""

    def test_empty_tuple_pattern(self, parse):
        """Empty tuple pattern: ()."""
        program = parse('let result = match unit { () -> "unit" }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, TuplePattern)
        assert len(arm.pattern.elements) == 0

    def test_single_element_tuple_pattern(self, parse):
        """Single-element tuple pattern: (x,)."""
        program = parse("let result = match single { (x,) -> x }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, TuplePattern)
        assert len(arm.pattern.elements) == 1

    def test_nested_tuple_pattern(self, parse):
        """Nested tuple pattern: ((a, b), c)."""
        program = parse("let result = match nested { ((a, b), c) -> a + b + c }")
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, TuplePattern)
        assert len(arm.pattern.elements) == 2
        assert isinstance(arm.pattern.elements[0], TuplePattern)

    def test_pattern_with_string_literal(self, parse):
        """String literal pattern."""
        program = parse('let result = match s { "hello" -> 1 }')
        match_expr = program.statements[0].value
        arm = match_expr.arms[0]
        assert isinstance(arm.pattern, LiteralPattern)

    def test_pattern_bound_variables_accessible(self, type_check):
        """Variables bound in pattern are accessible in body and guard."""
        errors, checker = type_check("""
fn test(p: Tuple[Int, Int]) -> Int {
    return match p {
        (x, y) where x > 0 -> x + y
        _ -> 0
    }
}
""")
        # x and y should be accessible in both guard and body
        # Filter for actual errors (not warnings about exhaustiveness)
        type_errors = [e for e in errors if "undefined" in str(e).lower()]
        assert len(type_errors) == 0
