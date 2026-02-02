"""
Unit tests for MathViz comprehensions and iterator methods.

Tests cover:
- List comprehensions
- Set comprehensions
- Dict comprehensions
- Pipe lambdas
- Iterator method code generation
"""

import pytest

from mathviz.compiler.ast_nodes import (
    ListComprehension,
    SetComprehension,
    DictComprehension,
    PipeLambda,
    ComprehensionClause,
    Identifier,
    BinaryExpression,
    RangeExpression,
    IntegerLiteral,
    LetStatement,
    ListLiteral,
)


class TestListComprehensionParsing:
    """Tests for list comprehension parsing."""

    def test_simple_list_comprehension(self, parse):
        """Test parsing a simple list comprehension."""
        ast = parse("let squares = [x for x in 0..10]")
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, ListComprehension)
        comp = stmt.value
        assert isinstance(comp.element, Identifier)
        assert comp.element.name == "x"
        assert len(comp.clauses) == 1
        assert comp.clauses[0].variable == "x"

    def test_list_comprehension_with_expression(self, parse):
        """Test list comprehension with transformed element."""
        ast = parse("let squares = [x^2 for x in 0..10]")
        stmt = ast.statements[0]
        comp = stmt.value
        assert isinstance(comp, ListComprehension)
        assert isinstance(comp.element, BinaryExpression)

    def test_list_comprehension_with_filter(self, parse):
        """Test list comprehension with if condition."""
        ast = parse("let evens = [x for x in 0..100 if x % 2 == 0]")
        stmt = ast.statements[0]
        comp = stmt.value
        assert isinstance(comp, ListComprehension)
        assert len(comp.clauses) == 1
        assert comp.clauses[0].condition is not None

    def test_nested_list_comprehension(self, parse):
        """Test list comprehension with multiple for clauses."""
        ast = parse("let pairs = [(x, y) for x in 0..3 for y in 0..3]")
        stmt = ast.statements[0]
        comp = stmt.value
        assert isinstance(comp, ListComprehension)
        assert len(comp.clauses) == 2
        assert comp.clauses[0].variable == "x"
        assert comp.clauses[1].variable == "y"

    def test_regular_list_literal(self, parse):
        """Ensure regular list literals still work."""
        ast = parse("let nums = [1, 2, 3]")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ListLiteral)
        assert len(stmt.value.elements) == 3

    def test_empty_list(self, parse):
        """Ensure empty list literal still works."""
        ast = parse("let empty = []")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ListLiteral)
        assert len(stmt.value.elements) == 0


class TestSetComprehensionParsing:
    """Tests for set comprehension parsing."""

    def test_simple_set_comprehension(self, parse):
        """Test parsing a simple set comprehension."""
        ast = parse("let unique = {x % 10 for x in 0..100}")
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, SetComprehension)

    def test_set_comprehension_with_filter(self, parse):
        """Test set comprehension with if condition."""
        ast = parse("let odds = {x for x in 0..50 if x % 2 == 1}")
        stmt = ast.statements[0]
        comp = stmt.value
        assert isinstance(comp, SetComprehension)
        assert comp.clauses[0].condition is not None


class TestDictComprehensionParsing:
    """Tests for dict comprehension parsing."""

    def test_simple_dict_comprehension(self, parse):
        """Test parsing a simple dict comprehension."""
        ast = parse("let squared = {x: x^2 for x in 0..10}")
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, DictComprehension)
        comp = stmt.value
        assert isinstance(comp.key, Identifier)
        assert isinstance(comp.value, BinaryExpression)

    def test_dict_comprehension_with_filter(self, parse):
        """Test dict comprehension with if condition."""
        ast = parse("let even_sq = {x: x^2 for x in 0..10 if x % 2 == 0}")
        stmt = ast.statements[0]
        comp = stmt.value
        assert isinstance(comp, DictComprehension)
        assert comp.clauses[0].condition is not None


class TestPipeLambdaParsing:
    """Tests for pipe-style lambda parsing."""

    def test_single_param_pipe_lambda(self, parse):
        """Test parsing single parameter pipe lambda."""
        ast = parse("let double = |x| x * 2")
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, PipeLambda)
        lam = stmt.value
        assert lam.parameters == ("x",)
        assert isinstance(lam.body, BinaryExpression)

    def test_multi_param_pipe_lambda(self, parse):
        """Test parsing multi-parameter pipe lambda."""
        ast = parse("let add = |x, y| x + y")
        stmt = ast.statements[0]
        lam = stmt.value
        assert isinstance(lam, PipeLambda)
        assert lam.parameters == ("x", "y")

    def test_pipe_lambda_in_expression(self, parse):
        """Test pipe lambda used in expression context."""
        ast = parse("let result = numbers.map(|x| x * 2)")
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)


class TestComprehensionCodeGen:
    """Tests for comprehension code generation."""

    def test_list_comprehension_codegen(self, compile_source):
        """Test list comprehension generates valid Python."""
        code = compile_source("let squares = [x^2 for x in 0..10]")
        # Check for key elements (parentheses may vary)
        assert "x ** 2" in code
        assert "for x in range(0, 10)" in code
        assert "squares =" in code

    def test_list_comprehension_with_filter_codegen(self, compile_source):
        """Test list comprehension with filter generates valid Python."""
        code = compile_source("let evens = [x for x in 0..100 if x % 2 == 0]")
        assert "for x in range(0, 100)" in code
        assert "x % 2" in code
        assert "if" in code

    def test_set_comprehension_codegen(self, compile_source):
        """Test set comprehension generates valid Python."""
        code = compile_source("let unique = {x % 10 for x in 0..100}")
        assert "x % 10" in code
        assert "for x in range(0, 100)" in code
        assert "unique =" in code

    def test_dict_comprehension_codegen(self, compile_source):
        """Test dict comprehension generates valid Python."""
        code = compile_source("let squared = {x: x^2 for x in 0..10}")
        assert "x:" in code
        assert "x ** 2" in code
        assert "for x in range(0, 10)" in code

    def test_nested_comprehension_codegen(self, compile_source):
        """Test nested comprehension generates valid Python."""
        code = compile_source("let pairs = [(x, y) for x in 0..3 for y in 0..3]")
        assert "for x in range(0, 3) for y in range(0, 3)" in code

    def test_pipe_lambda_codegen(self, compile_source):
        """Test pipe lambda generates valid Python lambda."""
        code = compile_source("let double = |x| x * 2")
        assert "lambda x:" in code
        assert "x * 2" in code

    def test_multi_param_pipe_lambda_codegen(self, compile_source):
        """Test multi-param pipe lambda generates valid Python."""
        code = compile_source("let add = |x, y| x + y")
        assert "lambda x, y:" in code
        assert "x + y" in code


class TestIteratorMethodCodeGen:
    """Tests for iterator method code generation."""

    def test_map_method_codegen(self, compile_source):
        """Test map method generates iter_map call."""
        code = compile_source("""
let numbers = [1, 2, 3]
let doubled = numbers.map(|x| x * 2)
""")
        assert "iter_map" in code
        assert "mathviz.runtime.iterators" in code

    def test_filter_method_codegen(self, compile_source):
        """Test filter method generates iter_filter call."""
        code = compile_source("""
let numbers = [1, 2, 3, 4]
let evens = numbers.filter(|x| x % 2 == 0)
""")
        assert "iter_filter" in code

    def test_reduce_method_codegen(self, compile_source):
        """Test reduce method generates iter_reduce call."""
        code = compile_source("""
let numbers = [1, 2, 3, 4]
let total = numbers.reduce(0, |acc, x| acc + x)
""")
        assert "iter_reduce" in code

    def test_first_method_codegen(self, compile_source):
        """Test first method generates iter_first call."""
        code = compile_source("""
let numbers = [1, 2, 3]
let first = numbers.first()
""")
        assert "iter_first" in code

    def test_sum_method_codegen(self, compile_source):
        """Test sum method generates iter_sum call."""
        code = compile_source("""
let numbers = [1, 2, 3, 4]
let total = numbers.sum()
""")
        assert "iter_sum" in code

    def test_take_method_codegen(self, compile_source):
        """Test take method generates iter_take call."""
        code = compile_source("""
let numbers = [1, 2, 3, 4, 5]
let first3 = numbers.take(3)
""")
        assert "iter_take" in code

    def test_sorted_method_codegen(self, compile_source):
        """Test sorted method generates iter_sorted call."""
        code = compile_source("""
let numbers = [3, 1, 4, 1, 5]
let sorted_nums = numbers.sorted()
""")
        assert "iter_sorted" in code

    def test_chained_methods_codegen(self, compile_source):
        """Test chained iterator methods generate correct code."""
        code = compile_source("""
let numbers = [1, 2, 3, 4, 5]
let result = numbers.filter(|x| x > 2).map(|x| x * 2)
""")
        assert "iter_filter" in code
        assert "iter_map" in code

    def test_unique_method_codegen(self, compile_source):
        """Test unique method generates iter_unique call."""
        code = compile_source("""
let numbers = [1, 2, 2, 3, 1]
let uniq = numbers.unique()
""")
        assert "iter_unique" in code

    def test_enumerate_method_codegen(self, compile_source):
        """Test enumerate method generates iter_enumerate call."""
        code = compile_source("""
let items = [10, 20, 30]
let indexed = items.enumerate()
""")
        assert "iter_enumerate" in code
