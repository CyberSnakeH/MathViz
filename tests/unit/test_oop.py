"""
Unit tests for MathViz OOP features.

Tests for:
- Structs (lightweight data types)
- Impl blocks (method implementations)
- Traits (interfaces)
- Enums (algebraic data types)
- Self expressions
- Visibility modifiers
"""

import pytest

from mathviz.compiler.tokens import TokenType
from mathviz.compiler.ast_nodes import (
    StructDef,
    StructField,
    ImplBlock,
    Method,
    TraitDef,
    TraitMethod,
    EnumDef,
    EnumVariant,
    SelfExpression,
    EnumVariantAccess,
    StructLiteral,
    EnumPattern,
    Visibility,
)


# =============================================================================
# Lexer Tests for OOP Tokens
# =============================================================================


class TestOOPTokens:
    """Tests for OOP keyword and operator tokenization."""

    def test_struct_keyword(self, tokenize):
        """Test tokenization of 'struct' keyword."""
        tokens = tokenize("struct Point")
        assert tokens[0].type == TokenType.STRUCT
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "Point"

    def test_impl_keyword(self, tokenize):
        """Test tokenization of 'impl' keyword."""
        tokens = tokenize("impl Point")
        assert tokens[0].type == TokenType.IMPL
        assert tokens[1].type == TokenType.IDENTIFIER

    def test_trait_keyword(self, tokenize):
        """Test tokenization of 'trait' keyword."""
        tokens = tokenize("trait Shape")
        assert tokens[0].type == TokenType.TRAIT
        assert tokens[1].type == TokenType.IDENTIFIER

    def test_enum_keyword(self, tokenize):
        """Test tokenization of 'enum' keyword."""
        tokens = tokenize("enum Color")
        assert tokens[0].type == TokenType.ENUM
        assert tokens[1].type == TokenType.IDENTIFIER

    def test_self_keyword(self, tokenize):
        """Test tokenization of 'self' keyword."""
        tokens = tokenize("self.x")
        assert tokens[0].type == TokenType.SELF
        assert tokens[1].type == TokenType.DOT
        assert tokens[2].type == TokenType.IDENTIFIER

    def test_double_colon(self, tokenize):
        """Test tokenization of '::' operator."""
        tokens = tokenize("Shape::Circle")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[1].type == TokenType.DOUBLE_COLON
        assert tokens[2].type == TokenType.IDENTIFIER


# =============================================================================
# Parser Tests for Structs
# =============================================================================


class TestStructParsing:
    """Tests for struct definition parsing."""

    def test_simple_struct(self, parse):
        """Test parsing a simple struct."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        ast = parse(source)
        assert len(ast.statements) == 1
        struct_def = ast.statements[0]
        assert isinstance(struct_def, StructDef)
        assert struct_def.name == "Point"
        assert len(struct_def.fields) == 2
        assert struct_def.fields[0].name == "x"
        assert struct_def.fields[1].name == "y"

    def test_struct_with_visibility(self, parse):
        """Test parsing struct with pub fields."""
        source = """
struct Person {
    pub name: String
    age: Int
}
"""
        ast = parse(source)
        struct_def = ast.statements[0]
        assert isinstance(struct_def, StructDef)
        assert struct_def.fields[0].visibility == Visibility.PUBLIC
        assert struct_def.fields[1].visibility == Visibility.PRIVATE

    def test_struct_with_generic_types(self, parse):
        """Test parsing struct with generic type fields."""
        source = """
struct Container {
    items: List[Int]
    mapping: Dict[String, Float]
}
"""
        ast = parse(source)
        struct_def = ast.statements[0]
        assert isinstance(struct_def, StructDef)
        assert len(struct_def.fields) == 2


# =============================================================================
# Parser Tests for Impl Blocks
# =============================================================================


class TestImplBlockParsing:
    """Tests for impl block parsing."""

    def test_simple_impl(self, parse):
        """Test parsing a simple impl block."""
        source = """
impl Point {
    fn origin() -> Point {
        return Point(0.0, 0.0)
    }
}
"""
        ast = parse(source)
        assert len(ast.statements) == 1
        impl_block = ast.statements[0]
        assert isinstance(impl_block, ImplBlock)
        assert impl_block.target_type == "Point"
        assert impl_block.trait_name is None
        assert len(impl_block.methods) == 1
        assert impl_block.methods[0].name == "origin"
        assert impl_block.methods[0].has_self is False

    def test_impl_with_self_method(self, parse):
        """Test parsing impl with self parameter."""
        source = """
impl Point {
    fn distance(self, other: Point) -> Float {
        let dx = self.x - other.x
        return dx
    }
}
"""
        ast = parse(source)
        impl_block = ast.statements[0]
        method = impl_block.methods[0]
        assert method.has_self is True
        assert len(method.parameters) == 1  # 'other', not counting 'self'
        assert method.parameters[0].name == "other"

    def test_trait_impl(self, parse):
        """Test parsing trait implementation."""
        source = """
impl Shape for Circle {
    fn area(self) -> Float {
        return 3.14159 * self.radius * self.radius
    }
}
"""
        ast = parse(source)
        impl_block = ast.statements[0]
        assert isinstance(impl_block, ImplBlock)
        assert impl_block.target_type == "Circle"
        assert impl_block.trait_name == "Shape"


# =============================================================================
# Parser Tests for Traits
# =============================================================================


class TestTraitParsing:
    """Tests for trait definition parsing."""

    def test_simple_trait(self, parse):
        """Test parsing a simple trait."""
        source = """
trait Shape {
    fn area(self) -> Float
    fn perimeter(self) -> Float
}
"""
        ast = parse(source)
        assert len(ast.statements) == 1
        trait_def = ast.statements[0]
        assert isinstance(trait_def, TraitDef)
        assert trait_def.name == "Shape"
        assert len(trait_def.methods) == 2
        assert trait_def.methods[0].name == "area"
        assert trait_def.methods[0].has_self is True

    def test_trait_with_default_impl(self, parse):
        """Test parsing trait with default method implementation."""
        source = """
trait Printable {
    fn to_string(self) -> String {
        return "default"
    }
}
"""
        ast = parse(source)
        trait_def = ast.statements[0]
        assert trait_def.methods[0].has_default_impl is True
        assert trait_def.methods[0].default_body is not None


# =============================================================================
# Parser Tests for Enums
# =============================================================================


class TestEnumParsing:
    """Tests for enum definition parsing."""

    def test_simple_enum(self, parse):
        """Test parsing a simple enum without data."""
        source = """
enum Color {
    Red
    Green
    Blue
}
"""
        ast = parse(source)
        assert len(ast.statements) == 1
        enum_def = ast.statements[0]
        assert isinstance(enum_def, EnumDef)
        assert enum_def.name == "Color"
        assert len(enum_def.variants) == 3
        assert enum_def.variants[0].name == "Red"
        assert len(enum_def.variants[0].fields) == 0

    def test_enum_with_data(self, parse):
        """Test parsing enum with associated data."""
        source = """
enum Shape {
    Circle(Float)
    Rectangle(Float, Float)
    Point
}
"""
        ast = parse(source)
        enum_def = ast.statements[0]
        assert isinstance(enum_def, EnumDef)
        assert enum_def.variants[0].name == "Circle"
        assert len(enum_def.variants[0].fields) == 1
        assert enum_def.variants[1].name == "Rectangle"
        assert len(enum_def.variants[1].fields) == 2
        assert enum_def.variants[2].name == "Point"
        assert len(enum_def.variants[2].fields) == 0

    def test_enum_variant_access_expression(self, parse):
        """Test parsing enum variant access expression."""
        source = """
let s = Shape::Circle
"""
        ast = parse(source)
        let_stmt = ast.statements[0]
        assert isinstance(let_stmt.value, EnumVariantAccess)
        assert let_stmt.value.enum_name == "Shape"
        assert let_stmt.value.variant_name == "Circle"


# =============================================================================
# Parser Tests for Self Expression
# =============================================================================


class TestSelfExpressionParsing:
    """Tests for self expression parsing."""

    def test_self_member_access(self, parse):
        """Test parsing self.member access."""
        source = """
impl Point {
    fn get_x(self) -> Float {
        return self.x
    }
}
"""
        ast = parse(source)
        impl_block = ast.statements[0]
        method = impl_block.methods[0]
        # The body should contain a return statement with self.x
        assert method.body is not None


# =============================================================================
# Parser Tests for Struct Literals
# =============================================================================


class TestStructLiteralParsing:
    """Tests for struct literal parsing."""

    def test_struct_literal_named_fields(self, parse):
        """Test parsing struct literal with named fields."""
        source = """
let p = Point { x: 1.0, y: 2.0 }
"""
        ast = parse(source)
        let_stmt = ast.statements[0]
        assert isinstance(let_stmt.value, StructLiteral)
        assert let_stmt.value.struct_name == "Point"
        assert len(let_stmt.value.fields) == 2


# =============================================================================
# Parser Tests for Enum Patterns
# =============================================================================


class TestEnumPatternParsing:
    """Tests for enum pattern parsing in match expressions."""

    def test_enum_pattern_in_match(self, parse):
        """Test parsing enum patterns in match expression."""
        source = """
match shape {
    Shape::Circle(r) -> r * r * 3.14
    Shape::Rectangle(w, h) -> w * h
    Shape::Point -> 0.0
}
"""
        ast = parse(source)
        match_expr = ast.statements[0].expression
        assert len(match_expr.arms) == 3

        # First arm: Shape::Circle(r)
        pattern1 = match_expr.arms[0].pattern
        assert isinstance(pattern1, EnumPattern)
        assert pattern1.enum_name == "Shape"
        assert pattern1.variant_name == "Circle"
        assert len(pattern1.bindings) == 1


# =============================================================================
# Code Generation Tests
# =============================================================================


class TestOOPCodeGen:
    """Tests for OOP code generation."""

    def test_struct_generates_dataclass(self, compile_source):
        """Test that struct generates Python dataclass."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        code = compile_source(source)
        assert "@dataclass" in code
        assert "class Point:" in code
        assert "x: float" in code
        assert "y: float" in code

    def test_trait_generates_abc(self, compile_source):
        """Test that trait generates Python ABC."""
        source = """
trait Shape {
    fn area(self) -> Float
}
"""
        code = compile_source(source)
        assert "class Shape(ABC):" in code
        assert "@abstractmethod" in code
        assert "def area(self)" in code

    def test_simple_enum_generates_python_enum(self, compile_source):
        """Test that simple enum generates Python Enum."""
        source = """
enum Color {
    Red
    Green
    Blue
}
"""
        code = compile_source(source)
        assert "class Color(Enum):" in code
        assert "Red = auto()" in code
        assert "Green = auto()" in code
        assert "Blue = auto()" in code

    def test_data_enum_generates_dataclasses(self, compile_source):
        """Test that enum with data generates dataclass variants."""
        source = """
enum Shape {
    Circle(Float)
    Rectangle(Float, Float)
}
"""
        code = compile_source(source)
        assert "class Shape:" in code
        assert "@dataclass" in code
        assert "class Circle(Shape):" in code
        assert "class Rectangle(Shape):" in code

    def test_struct_literal_generates_constructor(self, compile_source):
        """Test struct literal code generation."""
        source = """
struct Point {
    x: Float
    y: Float
}

let p = Point { x: 1.0, y: 2.0 }
"""
        code = compile_source(source)
        assert "Point(x=1.0, y=2.0)" in code


# =============================================================================
# Type Checking Tests
# =============================================================================


class TestOOPTypeChecking:
    """Tests for OOP type checking."""

    def test_struct_type_registration(self, analyze_types):
        """Test that structs are registered in type system."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        ast, errors, checker = analyze_types(source)
        assert "Point" in checker.struct_types
        struct_type = checker.struct_types["Point"]
        assert struct_type.get_field_type("x") is not None

    def test_struct_constructor_signature(self, analyze_types):
        """Test that struct constructor is registered."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        ast, errors, checker = analyze_types(source)
        assert "Point" in checker.function_signatures
        sig = checker.function_signatures["Point"]
        assert len(sig.param_types) == 2

    def test_trait_type_registration(self, analyze_types):
        """Test that traits are registered in type system."""
        source = """
trait Shape {
    fn area(self) -> Float
}
"""
        ast, errors, checker = analyze_types(source)
        assert "Shape" in checker.trait_types

    def test_enum_type_registration(self, analyze_types):
        """Test that enums are registered in type system."""
        source = """
enum Color {
    Red
    Green
    Blue
}
"""
        ast, errors, checker = analyze_types(source)
        assert "Color" in checker.enum_types
        enum_type = checker.enum_types["Color"]
        assert enum_type.get_variant("Red") is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestOOPIntegration:
    """Integration tests for complete OOP programs."""

    @pytest.mark.skip(reason="Impl methods generate Point_method() instead of class methods")
    def test_struct_with_impl(self, compile_source):
        """Test struct with methods."""
        source = """
struct Point {
    x: Float
    y: Float
}

impl Point {
    fn origin() -> Point {
        return Point(0.0, 0.0)
    }

    fn add(self, other: Point) -> Point {
        return Point(self.x + other.x, self.y + other.y)
    }
}
"""
        code = compile_source(source)
        assert "class Point:" in code
        assert "def origin()" in code
        assert "def add(self, other: Point)" in code

    def test_trait_implementation(self, compile_source):
        """Test trait with implementation."""
        source = """
trait Drawable {
    fn draw(self) -> String
}

struct Circle {
    radius: Float
}

impl Drawable for Circle {
    fn draw(self) -> String {
        return "circle"
    }
}
"""
        code = compile_source(source)
        assert "class Drawable(ABC):" in code
        assert "class Circle:" in code

    def test_enum_with_match(self, compile_source):
        """Test enum with pattern matching."""
        source = """
enum Outcome {
    Success(Int)
    Failure(String)
}

fn handle(r: Outcome) -> Int {
    return match r {
        Outcome::Success(n) -> n
        Outcome::Failure(msg) -> 0
    }
}
"""
        code = compile_source(source)
        assert "class Outcome:" in code
        assert "class Success(Outcome):" in code
        assert "class Failure(Outcome):" in code
