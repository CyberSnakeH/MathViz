"""
Unit tests for MathViz Generics System.

Tests for:
- Generic functions with type parameters
- Generic structs (Box<T>, Pair<A, B>)
- Generic enums (Option<T>, Result<T, E>)
- Generic traits (Container<T>)
- Generic impl blocks
- Type parameter bounds
- Where clauses
- Code generation with TypeVar and Generic
"""

import pytest

from mathviz.compiler.tokens import TokenType
from mathviz.compiler.ast_nodes import (
    FunctionDef,
    StructDef,
    EnumDef,
    TraitDef,
    ImplBlock,
    Method,
    TypeParameter,
    WhereClause,
    SimpleType,
    GenericType,
)


# =============================================================================
# Lexer Tests for Generic Syntax
# =============================================================================


class TestGenericTokens:
    """Tests for generic-related token handling."""

    def test_angle_brackets(self, tokenize):
        """Test tokenization of angle brackets for generics."""
        tokens = tokenize("Box<T>")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "Box"
        assert tokens[1].type == TokenType.LT
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "T"
        assert tokens[3].type == TokenType.GT

    def test_multiple_type_params(self, tokenize):
        """Test tokenization of multiple type parameters."""
        tokens = tokenize("Pair<A, B>")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[1].type == TokenType.LT
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "A"
        assert tokens[3].type == TokenType.COMMA
        assert tokens[4].type == TokenType.IDENTIFIER
        assert tokens[4].value == "B"
        assert tokens[5].type == TokenType.GT

    def test_type_bounds_with_plus(self, tokenize):
        """Test tokenization of type bounds with + operator."""
        tokens = tokenize("T: Display + Clone")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "T"
        assert tokens[1].type == TokenType.COLON
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "Display"
        assert tokens[3].type == TokenType.PLUS
        assert tokens[4].type == TokenType.IDENTIFIER
        assert tokens[4].value == "Clone"

    def test_where_keyword(self, tokenize):
        """Test tokenization of 'where' keyword."""
        tokens = tokenize("where T: Display")
        assert tokens[0].type == TokenType.WHERE
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[2].type == TokenType.COLON


# =============================================================================
# Parser Tests for Generic Functions
# =============================================================================


class TestGenericFunctionParsing:
    """Tests for generic function parsing."""

    def test_simple_generic_function(self, parse):
        """Test parsing a generic function with one type parameter."""
        source = """
fn identity<T>(x: T) -> T {
    return x
}
"""
        ast = parse(source)
        assert len(ast.statements) == 1
        func = ast.statements[0]
        assert isinstance(func, FunctionDef)
        assert func.name == "identity"
        assert len(func.type_params) == 1
        assert func.type_params[0].name == "T"
        assert func.type_params[0].bounds == ()

    def test_multiple_type_parameters(self, parse):
        """Test parsing a function with multiple type parameters."""
        source = """
fn swap<T, U>(a: T, b: U) -> T {
    return a
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert isinstance(func, FunctionDef)
        assert len(func.type_params) == 2
        assert func.type_params[0].name == "T"
        assert func.type_params[1].name == "U"

    def test_type_parameter_with_bound(self, parse):
        """Test parsing a type parameter with a single bound."""
        source = """
fn print_all<T: Display>(items: List[T]) {
    for item in items {
        println("{}", item)
    }
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert len(func.type_params) == 1
        assert func.type_params[0].name == "T"
        assert func.type_params[0].bounds == ("Display",)

    def test_type_parameter_with_multiple_bounds(self, parse):
        """Test parsing a type parameter with multiple bounds."""
        source = """
fn process<T: Display + Clone>(item: T) {
    let copy = item.clone()
    println("{}", copy)
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert len(func.type_params) == 1
        assert func.type_params[0].name == "T"
        assert func.type_params[0].bounds == ("Display", "Clone")

    def test_where_clause(self, parse):
        """Test parsing a function with a where clause."""
        source = """
fn compare<T>(a: T, b: T) -> Int where T: Ord {
    if a < b { return -1 }
    if a > b { return 1 }
    return 0
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert func.where_clause is not None
        assert len(func.where_clause.constraints) == 1
        assert func.where_clause.constraints[0][0] == "T"
        assert func.where_clause.constraints[0][1] == ("Ord",)


# =============================================================================
# Parser Tests for Generic Structs
# =============================================================================


class TestGenericStructParsing:
    """Tests for generic struct parsing."""

    def test_simple_generic_struct(self, parse):
        """Test parsing a generic struct with one type parameter."""
        source = """
struct Box<T> {
    value: T
}
"""
        ast = parse(source)
        assert len(ast.statements) == 1
        struct = ast.statements[0]
        assert isinstance(struct, StructDef)
        assert struct.name == "Box"
        assert len(struct.type_params) == 1
        assert struct.type_params[0].name == "T"
        assert len(struct.fields) == 1
        assert struct.fields[0].name == "value"

    def test_multiple_type_parameter_struct(self, parse):
        """Test parsing a struct with multiple type parameters."""
        source = """
struct Pair<A, B> {
    first: A
    second: B
}
"""
        ast = parse(source)
        struct = ast.statements[0]
        assert len(struct.type_params) == 2
        assert struct.type_params[0].name == "A"
        assert struct.type_params[1].name == "B"
        assert len(struct.fields) == 2

    def test_nested_generic_types(self, parse):
        """Test parsing a struct with nested generic types."""
        source = """
struct Container<T> {
    items: List[T]
    cache: Dict[String, T]
}
"""
        ast = parse(source)
        struct = ast.statements[0]
        assert len(struct.type_params) == 1
        assert len(struct.fields) == 2


# =============================================================================
# Parser Tests for Generic Enums
# =============================================================================


class TestGenericEnumParsing:
    """Tests for generic enum parsing."""

    def test_option_enum(self, parse):
        """Test parsing the Option<T> enum."""
        source = """
enum MyOption<T> {
    SomeValue(T)
    NoValue
}
"""
        ast = parse(source)
        enum = ast.statements[0]
        assert isinstance(enum, EnumDef)
        assert enum.name == "MyOption"
        assert len(enum.type_params) == 1
        assert enum.type_params[0].name == "T"
        assert len(enum.variants) == 2
        assert enum.variants[0].name == "SomeValue"
        assert enum.variants[1].name == "NoValue"

    def test_result_enum(self, parse):
        """Test parsing the Result<T, E> enum."""
        source = """
enum MyResult<T, E> {
    OkValue(T)
    ErrValue(E)
}
"""
        ast = parse(source)
        enum = ast.statements[0]
        assert enum.name == "MyResult"
        assert len(enum.type_params) == 2
        assert enum.type_params[0].name == "T"
        assert enum.type_params[1].name == "E"
        assert len(enum.variants) == 2


# =============================================================================
# Parser Tests for Generic Traits
# =============================================================================


class TestGenericTraitParsing:
    """Tests for generic trait parsing."""

    def test_generic_trait(self, parse):
        """Test parsing a generic trait."""
        source = """
trait Container<T> {
    fn get(self) -> T
    fn set(self, value: T)
}
"""
        ast = parse(source)
        trait = ast.statements[0]
        assert isinstance(trait, TraitDef)
        assert trait.name == "Container"
        assert len(trait.type_params) == 1
        assert trait.type_params[0].name == "T"
        assert len(trait.methods) == 2


# =============================================================================
# Parser Tests for Generic Impl Blocks
# =============================================================================


class TestGenericImplParsing:
    """Tests for generic impl block parsing."""

    def test_generic_impl_block(self, parse):
        """Test parsing a generic impl block."""
        source = """
struct Box<T> {
    value: T
}

impl<T> Box<T> {
    fn get_value(self) -> T {
        return self.value
    }

    fn set_value(self, v: T) {
        self.value = v
    }
}
"""
        ast = parse(source)
        assert len(ast.statements) == 2
        impl = ast.statements[1]
        assert isinstance(impl, ImplBlock)
        assert len(impl.type_params) == 1
        assert impl.type_params[0].name == "T"
        assert impl.target_type == "Box"
        assert impl.target_type_args == ("T",)

    def test_bounded_generic_impl(self, parse):
        """Test parsing a generic impl with bounds."""
        source = """
struct Box<T> {
    value: T
}

impl<T: Display> Box<T> {
    fn show(self) {
        print(self.value)
    }
}
"""
        ast = parse(source)
        impl = ast.statements[1]
        assert len(impl.type_params) == 1
        assert impl.type_params[0].name == "T"
        assert impl.type_params[0].bounds == ("Display",)


# =============================================================================
# Code Generation Tests for Generics
# =============================================================================


class TestGenericCodeGeneration:
    """Tests for generic code generation."""

    def test_generic_function_codegen(self, compile_to_python):
        """Test code generation for a generic function."""
        source = """
fn identity<T>(x: T) -> T {
    return x
}
"""
        code = compile_to_python(source)
        assert "TypeVar" in code
        assert "T = TypeVar('T')" in code
        assert "def identity(x: T) -> T:" in code

    def test_generic_struct_codegen(self, compile_to_python):
        """Test code generation for a generic struct."""
        source = """
struct Box<T> {
    value: T
}
"""
        code = compile_to_python(source)
        assert "TypeVar" in code
        assert "Generic" in code
        assert "T = TypeVar('T')" in code
        assert "class Box(Generic[T]):" in code
        assert "value: T" in code

    def test_generic_enum_codegen(self, compile_to_python):
        """Test code generation for a generic enum."""
        source = """
enum MyOption<T> {
    SomeValue(T)
    NoValue
}
"""
        code = compile_to_python(source)
        assert "TypeVar" in code
        assert "Generic" in code
        assert "T = TypeVar('T')" in code
        assert "class MyOption(Generic[T]):" in code
        assert "class SomeValue(MyOption[T]):" in code

    def test_generic_trait_codegen(self, compile_to_python):
        """Test code generation for a generic trait."""
        source = """
trait Container<T> {
    fn get(self) -> T
    fn set(self, value: T)
}
"""
        code = compile_to_python(source)
        assert "TypeVar" in code
        assert "Generic" in code
        assert "class Container(ABC, Generic[T]):" in code

    def test_multiple_type_params_codegen(self, compile_to_python):
        """Test code generation with multiple type parameters."""
        source = """
struct Pair<A, B> {
    first: A
    second: B
}
"""
        code = compile_to_python(source)
        assert "A = TypeVar('A')" in code
        assert "B = TypeVar('B')" in code
        assert "class Pair(Generic[A, B]):" in code


# =============================================================================
# Integration Tests
# =============================================================================


class TestGenericIntegration:
    """Integration tests for the complete generic pipeline."""

    def test_full_generic_pipeline(self, parse, compile_to_python):
        """Test the complete pipeline from parsing to code generation."""
        source = """
fn process<T, U>(value: T) -> U {
    return value
}

struct Box<T> {
    value: T
}

enum MyResult<T, E> {
    OkValue(T)
    ErrValue(E)
}
"""
        # First verify parsing works
        ast = parse(source)
        assert len(ast.statements) == 3

        func = ast.statements[0]
        assert isinstance(func, FunctionDef)
        assert len(func.type_params) == 2

        struct = ast.statements[1]
        assert isinstance(struct, StructDef)
        assert len(struct.type_params) == 1

        enum = ast.statements[2]
        assert isinstance(enum, EnumDef)
        assert len(enum.type_params) == 2

        # Then verify code generation
        code = compile_to_python(source)
        assert "T = TypeVar('T')" in code
        assert "U = TypeVar('U')" in code
        assert "E = TypeVar('E')" in code
        assert "def process(" in code
        assert "class Box(Generic[T]):" in code
        assert "class MyResult(Generic[T, E]):" in code

    def test_bounded_type_params_pipeline(self, parse, compile_to_python):
        """Test bounded type parameters through the full pipeline."""
        source = """
fn print_all<T: Display>(items: List[T]) {
    for item in items {
        println("{}", item)
    }
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert func.type_params[0].bounds == ("Display",)

        code = compile_to_python(source)
        # Note: In Python, we can't directly enforce bounds, but we declare them
        assert "T = TypeVar('T'" in code
        assert "def print_all(" in code


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestGenericEdgeCases:
    """Tests for edge cases in generic handling."""

    def test_non_generic_function(self, parse):
        """Test that non-generic functions still work."""
        source = """
fn add(x: Int, y: Int) -> Int {
    return x + y
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert func.type_params == ()

    def test_non_generic_struct(self, parse):
        """Test that non-generic structs still work."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        ast = parse(source)
        struct = ast.statements[0]
        assert struct.type_params == ()

    def test_generic_with_complex_return_type(self, parse):
        """Test generic function with list return type."""
        source = """
fn create_list<T>(value: T) -> List[T] {
    let result: List[T] = []
    result.push(value)
    return result
}
"""
        ast = parse(source)
        func = ast.statements[0]
        assert len(func.type_params) == 1
        # Return type should be a generic list type

    def test_nested_generics_in_struct(self, parse):
        """Test struct with nested generic types in fields."""
        source = """
struct Cache<K, V> {
    data: Dict[K, V]
    keys: List[K]
    values: List[V]
}
"""
        ast = parse(source)
        struct = ast.statements[0]
        assert len(struct.type_params) == 2
        assert len(struct.fields) == 3
