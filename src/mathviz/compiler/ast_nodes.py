"""
Abstract Syntax Tree (AST) node definitions for MathViz.

This module defines all AST node types representing the structure of
a MathViz program after parsing. Each node is immutable and carries
source location information for error reporting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union

from mathviz.utils.errors import SourceLocation


class ASTNode(ABC):
    """Base class for all AST nodes."""

    location: Optional[SourceLocation]

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor for tree traversal."""
        pass


class ASTVisitor(ABC):
    """
    Visitor pattern base class for AST traversal.

    Implement this to create custom AST processors (type checkers,
    code generators, optimizers, etc.).
    """

    def visit(self, node: ASTNode) -> Any:
        """Dispatch to the appropriate visit method."""
        return node.accept(self)


# -----------------------------------------------------------------------------
# Type Annotations
# -----------------------------------------------------------------------------


class TypeAnnotation(ASTNode):
    """Base class for type annotations."""

    pass


@dataclass(frozen=True, slots=True)
class SimpleType(TypeAnnotation):
    """
    A simple type annotation (Int, Float, Bool, String, etc.).

    Examples:
        Int, Float, Bool, String, MyClass
    """

    name: str
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_simple_type(self)


@dataclass(frozen=True, slots=True)
class GenericType(TypeAnnotation):
    """
    A generic type annotation.

    Examples:
        List[Int], Set[Float], Dict[String, Int]
    """

    base: str
    type_args: tuple[TypeAnnotation, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_generic_type(self)


@dataclass(frozen=True, slots=True)
class FunctionType(TypeAnnotation):
    """
    A function type annotation.

    Example:
        (Int, Int) -> Int
    """

    param_types: tuple[TypeAnnotation, ...]
    return_type: TypeAnnotation
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_type(self)


# -----------------------------------------------------------------------------
# Generic Type Parameters
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeParameter:
    """
    A generic type parameter with optional bounds.

    Examples:
        T                    - unbounded type parameter
        T: Display           - type parameter with single bound
        T: Display + Clone   - type parameter with multiple bounds

    Attributes:
        name: The name of the type parameter (e.g., "T", "U")
        bounds: Trait bounds that the type must implement
        location: Source location for error reporting
    """

    name: str
    bounds: tuple[str, ...] = ()
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class WhereClause:
    """
    A where clause for complex type constraints.

    Example:
        where T: Display, U: Clone + Debug

    Attributes:
        constraints: List of (type_param_name, bounds) pairs
        location: Source location for error reporting
    """

    constraints: tuple[tuple[str, tuple[str, ...]], ...]
    location: Optional[SourceLocation] = None


# -----------------------------------------------------------------------------
# Documentation Comments
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocComment:
    """
    A documentation comment attached to a definition.

    Documentation comments start with /// for single-line or /** */ for multi-line.
    They support Markdown formatting and special sections like # Examples.

    Examples:
        /// A simple doc comment
        ///
        /// # Examples
        /// ```
        /// let x = 42
        /// ```

        /**
         * Multi-line doc comment
         * with multiple lines
         */

    Attributes:
        content: The raw content of the documentation (with /// or * prefixes removed)
        location: Source location for error reporting
    """

    content: str
    location: Optional[SourceLocation] = None

    def get_summary(self) -> str:
        """Get the first line/paragraph as a summary."""
        lines = self.content.split("\n")
        summary_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("#"):
                break
            summary_lines.append(stripped)
        return " ".join(summary_lines)

    def get_section(self, section_name: str) -> str | None:
        """
        Get the content of a specific section (e.g., 'Examples', 'Arguments').

        Args:
            section_name: The name of the section (case-insensitive)

        Returns:
            The section content, or None if not found.
        """
        lines = self.content.split("\n")
        in_section = False
        section_lines: list[str] = []
        section_header = f"# {section_name}"

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(section_header.lower()):
                in_section = True
                continue
            if in_section:
                if stripped.startswith("# "):
                    break
                section_lines.append(line)

        if not section_lines:
            return None
        return "\n".join(section_lines).strip()

    def get_examples(self) -> list[str]:
        """
        Extract code examples from the doc comment.

        Returns:
            List of code snippets found in ``` blocks within # Examples section.
        """
        examples_section = self.get_section("Examples")
        if not examples_section:
            return []

        examples: list[str] = []
        lines = examples_section.split("\n")
        in_code_block = False
        code_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code_block:
                    # End of code block
                    examples.append("\n".join(code_lines))
                    code_lines = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                code_lines.append(line)

        return examples


# -----------------------------------------------------------------------------
# Attributes (Decorators)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attribute:
    """
    An attribute/decorator applied to a definition.

    Attributes are prefixed with @ and can have optional arguments.

    Examples:
        @test
        @test("description of the test")
        @should_panic
        @should_panic("expected panic message")
        @jit
        @njit(parallel=true)

    Attributes:
        name: The attribute name (e.g., "test", "should_panic", "jit")
        arguments: Tuple of expression arguments passed to the attribute
        location: Source location for error reporting
    """

    name: str
    arguments: tuple["Expression", ...] = ()
    location: Optional[SourceLocation] = None


# -----------------------------------------------------------------------------
# Expressions
# -----------------------------------------------------------------------------


class Expression(ASTNode):
    """Base class for all expressions."""

    pass


@dataclass(frozen=True, slots=True)
class Identifier(Expression):
    """
    An identifier expression.

    Example:
        x, myVariable, _private
    """

    name: str
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_identifier(self)


@dataclass(frozen=True, slots=True)
class IntegerLiteral(Expression):
    """An integer literal."""

    value: int
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_integer_literal(self)


@dataclass(frozen=True, slots=True)
class FloatLiteral(Expression):
    """A floating-point literal."""

    value: float
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_float_literal(self)


@dataclass(frozen=True, slots=True)
class StringLiteral(Expression):
    """A string literal."""

    value: str
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_string_literal(self)


# -----------------------------------------------------------------------------
# F-String (String Interpolation)
# -----------------------------------------------------------------------------


class FStringPart(ABC):
    """Base class for f-string parts (literal or expression)."""

    pass


@dataclass(frozen=True, slots=True)
class FStringLiteral(FStringPart):
    """
    Literal text part in an f-string.

    Example:
        In f"Hello, {name}!", "Hello, " is an FStringLiteral.
    """

    value: str


@dataclass(frozen=True, slots=True)
class FStringExpression(FStringPart):
    """
    Expression part in an f-string with optional format specifier.

    Example:
        In f"Value: {x:.2f}", the expression is x with format ".2f".
    """

    expression: "Expression"
    format_spec: Optional[str] = None  # e.g., ".2f", "05d", "x"


@dataclass(frozen=True, slots=True)
class FString(Expression):
    """
    An f-string (formatted string literal).

    Example:
        f"Hello, {name}!"
        f"Value: {x:.2f}"
        f"{a} + {b} = {a + b}"
    """

    parts: tuple[FStringPart, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_fstring(self)


@dataclass(frozen=True, slots=True)
class BooleanLiteral(Expression):
    """A boolean literal (true/false)."""

    value: bool
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_boolean_literal(self)


@dataclass(frozen=True, slots=True)
class NoneLiteral(Expression):
    """The None literal."""

    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_none_literal(self)


@dataclass(frozen=True, slots=True)
class ListLiteral(Expression):
    """
    A list literal expression.

    Example:
        [1, 2, 3]
    """

    elements: tuple[Expression, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_list_literal(self)


@dataclass(frozen=True, slots=True)
class SetLiteral(Expression):
    """
    A set literal expression.

    Example:
        {1, 2, 3}
    """

    elements: tuple[Expression, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_set_literal(self)


@dataclass(frozen=True, slots=True)
class DictLiteral(Expression):
    """
    A dictionary literal expression.

    Example:
        {"a": 1, "b": 2}
    """

    pairs: tuple[tuple[Expression, Expression], ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_dict_literal(self)


@dataclass(frozen=True, slots=True)
class TupleLiteral(Expression):
    """
    A tuple literal expression.

    Examples:
        (1, 2, 3)
        (1,)       # Single-element tuple
        ()         # Empty tuple
    """

    elements: tuple[Expression, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_tuple_literal(self)


@dataclass(frozen=True, slots=True)
class SomeExpression(Expression):
    """
    A Some(value) expression for Optional types.

    Example:
        Some(42)
    """

    value: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_some_expression(self)


@dataclass(frozen=True, slots=True)
class OkExpression(Expression):
    """
    An Ok(value) expression for Result types.

    Example:
        Ok(42)
    """

    value: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ok_expression(self)


@dataclass(frozen=True, slots=True)
class ErrExpression(Expression):
    """
    An Err(error) expression for Result types.

    Example:
        Err("Something went wrong")
    """

    value: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_err_expression(self)


@dataclass(frozen=True, slots=True)
class UnwrapExpression(Expression):
    """
    An unwrap expression using the ? operator.

    Propagates errors for Result types or unwraps Optional values.

    Examples:
        value?           # Propagate error or unwrap
        get_result()?    # Propagate error from function result
    """

    operand: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unwrap_expression(self)


@dataclass(frozen=True, slots=True)
class AwaitExpression(Expression):
    """
    An await expression for async operations.

    Awaits the result of an async function or Future.

    Examples:
        await fetch_data(url)
        await task
        let result = await http_get("https://api.example.com")
    """

    expression: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_await_expression(self)


class BinaryOperator(Enum):
    """Binary operator types."""

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOOR_DIV = auto()
    MOD = auto()
    POW = auto()

    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    # Logical
    AND = auto()
    OR = auto()

    # Set operations (mathematical)
    ELEMENT_OF = auto()       # ∈
    NOT_ELEMENT_OF = auto()   # ∉
    SUBSET = auto()           # ⊆
    SUPERSET = auto()         # ⊇
    PROPER_SUBSET = auto()    # ⊂
    PROPER_SUPERSET = auto()  # ⊃
    UNION = auto()            # ∪
    INTERSECTION = auto()     # ∩
    SET_DIFF = auto()         # ∖

    # Membership (Python-style)
    IN = auto()
    NOT_IN = auto()


class UnaryOperator(Enum):
    """Unary operator types."""

    NEG = auto()      # -
    NOT = auto()      # not
    POS = auto()      # +


@dataclass(frozen=True, slots=True)
class BinaryExpression(Expression):
    """
    A binary operation expression.

    Example:
        a + b, x ∈ S, A ∪ B
    """

    left: Expression
    operator: BinaryOperator
    right: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binary_expression(self)


@dataclass(frozen=True, slots=True)
class UnaryExpression(Expression):
    """
    A unary operation expression.

    Example:
        -x, not flag
    """

    operator: UnaryOperator
    operand: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary_expression(self)


@dataclass(frozen=True, slots=True)
class KeywordArgument(Expression):
    """
    A keyword/named argument in a function call.

    Example:
        Circle(radius: 1.0), func(x: 10, y: 20)
    """

    name: str
    value: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_keyword_argument(self)


@dataclass(frozen=True, slots=True)
class CallExpression(Expression):
    """
    A function/method call expression.

    Example:
        print("hello"), obj.method(x, y), Circle(radius: 1.0)
    """

    callee: Expression
    arguments: tuple[Expression, ...]
    keyword_arguments: tuple["KeywordArgument", ...] = ()
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_call_expression(self)


@dataclass(frozen=True, slots=True)
class MemberAccess(Expression):
    """
    A member access expression (dot notation).

    Example:
        obj.property, module.function
    """

    object: Expression
    member: str
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_member_access(self)


@dataclass(frozen=True, slots=True)
class IndexExpression(Expression):
    """
    An index/subscript expression.

    Example:
        arr[0], dict["key"]
    """

    object: Expression
    index: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_index_expression(self)


@dataclass(frozen=True, slots=True)
class ConditionalExpression(Expression):
    """
    A ternary conditional expression.

    Example:
        x if condition else y
    """

    condition: Expression
    then_expr: Expression
    else_expr: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_conditional_expression(self)


@dataclass(frozen=True, slots=True)
class LambdaExpression(Expression):
    """
    A lambda (anonymous function) expression.

    Example:
        (x, y) => x + y
    """

    parameters: tuple["Parameter", ...]
    body: Union[Expression, "Block"]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_lambda_expression(self)


@dataclass(frozen=True, slots=True)
class RangeExpression(Expression):
    """
    A range expression for iteration.

    Example:
        0..10, 1..=100
    """

    start: Expression
    end: Expression
    inclusive: bool = False
    step: Optional[Expression] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_range_expression(self)


# -----------------------------------------------------------------------------
# Comprehensions
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComprehensionClause:
    """
    A for...in clause in a comprehension, optionally with an if condition.

    Examples:
        for x in items
        for x in items if x > 0
        for (k, v) in dict.items()
    """

    variable: str
    iterable: Expression
    condition: Optional[Expression] = None  # Optional if clause
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class ListComprehension(Expression):
    """
    List comprehension: [expr for x in iter if cond].

    Examples:
        [x^2 for x in 0..10]
        [x for x in items if x % 2 == 0]
        [(x, y) for x in 0..3 for y in 0..3]
    """

    element: Expression
    clauses: tuple[ComprehensionClause, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_list_comprehension(self)


@dataclass(frozen=True, slots=True)
class SetComprehension(Expression):
    """
    Set comprehension: {expr for x in iter if cond}.

    Examples:
        {x % 10 for x in items}
        {x for x in items if x > 0}
    """

    element: Expression
    clauses: tuple[ComprehensionClause, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_set_comprehension(self)


@dataclass(frozen=True, slots=True)
class DictComprehension(Expression):
    """
    Dict comprehension: {k: v for x in iter if cond}.

    Examples:
        {name: score for (name, score) in results}
        {k: v^2 for (k, v) in dict}
        {x: x^2 for x in 0..10 if x % 2 == 0}
    """

    key: Expression
    value: Expression
    clauses: tuple[ComprehensionClause, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_dict_comprehension(self)


@dataclass(frozen=True, slots=True)
class PipeLambda(Expression):
    """
    Pipe-style lambda expression using |params| body syntax.

    Examples:
        |x| x * 2
        |x, y| x + y
        |acc, x| acc + x
    """

    parameters: tuple[str, ...]
    body: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_pipe_lambda(self)


# -----------------------------------------------------------------------------
# Pattern Matching
# -----------------------------------------------------------------------------


class Pattern(ABC):
    """
    Base class for match patterns.

    Patterns are used in match expressions to destructure and match values.
    """

    location: Optional[SourceLocation]

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor for pattern traversal."""
        pass

    @abstractmethod
    def get_bound_names(self) -> set[str]:
        """Return the set of variable names bound by this pattern."""
        pass


@dataclass(frozen=True, slots=True)
class LiteralPattern(Pattern):
    """
    Match a literal value: 0, "hello", true, None.

    Examples:
        0 -> println("zero")
        "hello" -> println("greeting")
        true -> println("yes")
    """

    value: Expression  # IntegerLiteral, FloatLiteral, StringLiteral, BooleanLiteral, NoneLiteral
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_literal_pattern(self)

    def get_bound_names(self) -> set[str]:
        return set()


@dataclass(frozen=True, slots=True)
class IdentifierPattern(Pattern):
    """
    Bind a value to a name: n, x, _.

    The wildcard pattern (_) matches anything but doesn't bind.

    Examples:
        n -> println("got {}", n)
        _ -> println("fallback")
    """

    name: str
    is_wildcard: bool = False
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_identifier_pattern(self)

    def get_bound_names(self) -> set[str]:
        if self.is_wildcard:
            return set()
        return {self.name}


@dataclass(frozen=True, slots=True)
class TuplePattern(Pattern):
    """
    Destructure a tuple: (x, y, z), (0, 0), (x, _).

    Examples:
        (0, 0) -> "origin"
        (x, 0) -> "on x-axis"
        (0, y) -> "on y-axis"
        (x, y) -> "general point"
    """

    elements: tuple[Pattern, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_tuple_pattern(self)

    def get_bound_names(self) -> set[str]:
        names: set[str] = set()
        for elem in self.elements:
            names.update(elem.get_bound_names())
        return names


@dataclass(frozen=True, slots=True)
class ConstructorPattern(Pattern):
    """
    Match a constructor: Circle(r), Some(x), Ok(value), Err(e).

    Examples:
        Circle(r) -> PI * r ^ 2
        Rectangle(w, h) -> w * h
        Some(x) -> x
        None -> 0
        Ok(value) -> value
        Err(e) -> panic(e)
    """

    name: str
    args: tuple[Pattern, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_constructor_pattern(self)

    def get_bound_names(self) -> set[str]:
        names: set[str] = set()
        for arg in self.args:
            names.update(arg.get_bound_names())
        return names


@dataclass(frozen=True, slots=True)
class RangePattern(Pattern):
    """
    Match a range of values: 1..10, 'a'..'z', 1..=10.

    Supports both exclusive (..) and inclusive (..=) ranges.

    Examples:
        0..12 -> "child"
        13..19 -> "teenager"
        'a'..'z' -> "lowercase"
        0..=100 -> "percentage"
    """

    start: Expression  # Inclusive start
    end: Expression    # Exclusive end (or inclusive with ..=)
    inclusive: bool = False  # True for ..= (inclusive end)
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_range_pattern(self)

    def get_bound_names(self) -> set[str]:
        return set()


@dataclass(frozen=True, slots=True)
class OrPattern(Pattern):
    """
    Match any of several patterns: a | b | c.

    All patterns in an OrPattern must bind the same set of variables.

    Examples:
        "Monday" | "Tuesday" | "Wednesday" -> "weekday"
        0 | 1 -> "binary"
        (0, 0) | (1, 1) -> "diagonal"
        (x, 0) | (0, x) -> "on axis"
    """

    patterns: tuple[Pattern, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_or_pattern(self)

    def get_bound_names(self) -> set[str]:
        # All patterns should bind the same names; return first's bindings
        if self.patterns:
            return self.patterns[0].get_bound_names()
        return set()


@dataclass(frozen=True, slots=True)
class BindingPattern(Pattern):
    """
    Bind a name while also matching a pattern: x @ pattern.

    Allows capturing the entire matched value while also destructuring.

    Examples:
        x @ 1..100 -> println("Got {} in range", x)
        x @ (a, b) -> println("Got tuple {} with {} and {}", x, a, b)
        list @ [first, ..rest] -> println("List {} starts with {}", list, first)
    """

    name: str
    pattern: Pattern
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binding_pattern(self)

    def get_bound_names(self) -> set[str]:
        names = {self.name}
        names.update(self.pattern.get_bound_names())
        return names


@dataclass(frozen=True, slots=True)
class RestPattern(Pattern):
    """
    Match rest of a sequence: ..rest or just ..

    Used in tuple and list patterns to match remaining elements.

    Examples:
        [first, ..rest] -> "has rest"
        [first, second, ..] -> "at least two"
        [.., last] -> "ends with last"
        (first, .., last) -> "first and last only"
    """

    name: Optional[str] = None  # None for anonymous rest (..)
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_rest_pattern(self)

    def get_bound_names(self) -> set[str]:
        if self.name:
            return {self.name}
        return set()


@dataclass(frozen=True, slots=True)
class ListPattern(Pattern):
    """
    Destructure a list/array: [x, y, z], [first, ..rest], [.., last].

    Similar to TuplePattern but for lists, and can contain RestPattern.

    Examples:
        [] -> "empty"
        [x] -> "single"
        [first, ..rest] -> f"starts with {first}"
        [first, second, ..] -> "at least two"
        [.., last] -> f"ends with {last}"
        [a, b, c] -> "exactly three"
    """

    elements: tuple[Pattern, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_list_pattern(self)

    def get_bound_names(self) -> set[str]:
        names: set[str] = set()
        for elem in self.elements:
            names.update(elem.get_bound_names())
        return names


@dataclass(frozen=True, slots=True)
class MatchArm:
    """
    Single arm in a match expression.

    Consists of a pattern, optional guard (where condition), and body.

    Example:
        n where n > 0 -> println("positive: {}", n)
    """

    pattern: Pattern
    guard: Optional[Expression]  # where condition
    body: Union[Expression, "Block"]
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class MatchExpression(Expression):
    """
    Pattern matching expression.

    Example:
        match value {
            0 -> println("zero")
            1 -> println("one")
            n where n > 0 -> println("positive: {}", n)
            n where n < 0 -> println("negative: {}", n)
            _ -> println("fallback")
        }
    """

    subject: Expression
    arms: tuple[MatchArm, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_match_expression(self)


# -----------------------------------------------------------------------------
# Statements
# -----------------------------------------------------------------------------


class Statement(ASTNode):
    """Base class for all statements."""

    pass


@dataclass(frozen=True, slots=True)
class ExpressionStatement(Statement):
    """
    An expression used as a statement.

    Example:
        print("hello")
    """

    expression: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_expression_statement(self)


@dataclass(frozen=True, slots=True)
class Parameter:
    """
    A function parameter definition.

    Example:
        x: Int, y: Float = 0.0
    """

    name: str
    type_annotation: Optional[TypeAnnotation] = None
    default_value: Optional[Expression] = None
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class Block(ASTNode):
    """
    A block of statements enclosed in braces.

    Example:
        { stmt1; stmt2; stmt3 }
    """

    statements: tuple[Statement, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_block(self)


@dataclass(frozen=True, slots=True)
class LetStatement(Statement):
    """
    A variable declaration statement.

    Example:
        let x: Int = 42
        let name = "Alice"
        let mut counter = 0
    """

    name: str
    type_annotation: Optional[TypeAnnotation] = None
    value: Optional[Expression] = None
    mutable: bool = False
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_let_statement(self)


@dataclass(frozen=True, slots=True)
class ConstDeclaration(Statement):
    """
    A compile-time constant declaration.

    Constants must have an initializer and are evaluated at compile time.
    They can be used in places requiring constant expressions.

    Example:
        const PI = 3.14159265358979
        const MAX_SIZE = 1024
        const TAU = 2.0 * PI
        const VERSION = "1.0.0"
    """

    name: str
    value: Expression  # Must be a constant expression
    type_annotation: Optional[TypeAnnotation] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_const_declaration(self)


@dataclass(frozen=True, slots=True)
class AssignmentStatement(Statement):
    """
    An assignment statement.

    Example:
        x = 10
        obj.prop = value
    """

    target: Expression
    value: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_assignment_statement(self)


@dataclass(frozen=True, slots=True)
class CompoundAssignment(Statement):
    """
    A compound assignment statement.

    Example:
        x += 10, y *= 2
    """

    target: Expression
    operator: BinaryOperator
    value: Expression
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_compound_assignment(self)


class JitMode(Enum):
    """JIT compilation modes for Numba optimization."""
    NONE = "none"           # No JIT compilation
    JIT = "jit"             # Standard @jit decorator
    NJIT = "njit"           # @njit (nopython=True, faster)
    VECTORIZE = "vectorize" # @vectorize for ufuncs
    GUVECTORIZE = "guvectorize"  # @guvectorize for generalized ufuncs
    STENCIL = "stencil"     # @stencil for stencil computations
    CFUNC = "cfunc"         # @cfunc for C-callable functions


@dataclass(frozen=True, slots=True)
class JitOptions:
    """
    Options for Numba JIT compilation.

    These map directly to Numba decorator arguments.
    """
    mode: JitMode = JitMode.NONE
    nopython: bool = True       # Compile in nopython mode (faster, stricter)
    nogil: bool = False         # Release GIL (for parallel execution)
    cache: bool = True          # Cache compiled function to disk
    parallel: bool = False      # Enable automatic parallelization
    fastmath: bool = False      # Use fast math (less precise, faster)
    boundscheck: bool = False   # Check array bounds (slower but safer)
    inline: str = "never"       # Inlining: "never", "always", or "recursive"

    def to_decorator_args(self) -> str:
        """Generate the decorator argument string."""
        args = []
        if self.mode in (JitMode.JIT, JitMode.NJIT):
            if self.nopython and self.mode == JitMode.JIT:
                args.append("nopython=True")
            if self.nogil:
                args.append("nogil=True")
            if self.cache:
                args.append("cache=True")
            if self.parallel:
                args.append("parallel=True")
            if self.fastmath:
                args.append("fastmath=True")
            if self.boundscheck:
                args.append("boundscheck=True")
            if self.inline != "never":
                args.append(f"inline='{self.inline}'")
        return ", ".join(args)


@dataclass(frozen=True, slots=True)
class FunctionDef(Statement):
    """
    A function definition with optional generics and Numba JIT optimization.

    Example:
        # Basic function
        fn add(x: Int, y: Int) -> Int {
            return x + y
        }

        # Generic function
        fn identity<T>(x: T) -> T {
            return x
        }

        # Generic function with bounds
        fn print_all<T: Display>(items: List[T]) {
            for item in items {
                println("{}", item)
            }
        }

        # Generic function with multiple type parameters
        fn map<T, U>(list: List[T], f: (T) -> U) -> List[U] {
            let result: List[U] = []
            for item in list {
                result.push(f(item))
            }
            return result
        }

        # JIT-optimized function
        @jit
        fn compute(arr: List[Float]) -> Float {
            let total = 0.0
            for x in arr {
                total += x * x
            }
            return total
        }
    """

    name: str
    parameters: tuple[Parameter, ...]
    return_type: Optional[TypeAnnotation] = None
    body: Optional[Block] = None
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    jit_options: JitOptions = field(default_factory=JitOptions)
    doc_comment: Optional["DocComment"] = None
    attributes: tuple["Attribute", ...] = ()
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this function has type parameters."""
        return len(self.type_params) > 0

    @property
    def is_optimized(self) -> bool:
        """Check if this function has JIT optimization enabled."""
        return self.jit_options.mode != JitMode.NONE

    @property
    def is_test(self) -> bool:
        """Check if this function is a test function."""
        return any(attr.name == "test" for attr in self.attributes)

    @property
    def should_panic(self) -> bool:
        """Check if this test function expects a panic/exception."""
        return any(attr.name == "should_panic" for attr in self.attributes)

    def get_test_description(self) -> str | None:
        """Get the test description if provided in @test attribute."""
        for attr in self.attributes:
            if attr.name == "test" and attr.arguments:
                from mathviz.compiler.ast_nodes import StringLiteral
                if isinstance(attr.arguments[0], StringLiteral):
                    return attr.arguments[0].value
        return None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_def(self)


@dataclass(frozen=True, slots=True)
class AsyncFunctionDef(Statement):
    """
    An async function definition.

    Async functions can contain await expressions and return Futures.

    Example:
        async fn fetch_data(url: String) -> Result[String, Error] {
            let response = await http_get(url)
            return Ok(response.body)
        }

        async fn main() {
            let data = await fetch_data("https://api.example.com")
            match data {
                Ok(body) -> println("Got: {}", body)
                Err(e) -> println("Error: {}", e)
            }
        }

        // Concurrent execution
        async fn fetch_all() {
            let (a, b, c) = await all(
                fetch_data("url1"),
                fetch_data("url2"),
                fetch_data("url3")
            )
        }
    """

    name: str
    parameters: tuple[Parameter, ...]
    return_type: Optional[TypeAnnotation] = None
    body: Optional[Block] = None
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this function has type parameters."""
        return len(self.type_params) > 0

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_async_function_def(self)


@dataclass(frozen=True, slots=True)
class ClassDef(Statement):
    """
    A class definition.

    Example:
        class Point {
            let x: Float
            let y: Float

            fn distance(other: Point) -> Float { ... }
        }
    """

    name: str
    base_classes: tuple[str, ...] = ()
    body: Optional[Block] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_class_def(self)


@dataclass(frozen=True, slots=True)
class SceneDef(Statement):
    """
    A Manim scene definition.

    Example:
        scene MyAnimation {
            fn construct() {
                let circle = Circle()
                self.play(Create(circle))
            }
        }
    """

    name: str
    body: Optional[Block] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_scene_def(self)


@dataclass(frozen=True, slots=True)
class IfStatement(Statement):
    """
    An if/elif/else statement.

    Example:
        if x > 0 {
            print("positive")
        } elif x < 0 {
            print("negative")
        } else {
            print("zero")
        }
    """

    condition: Expression
    then_block: Block
    elif_clauses: tuple[tuple[Expression, Block], ...] = ()
    else_block: Optional[Block] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_if_statement(self)


@dataclass(frozen=True, slots=True)
class ForStatement(Statement):
    """
    A for loop statement.

    Example:
        for i in 0..10 {
            print(i)
        }

        for x in items {
            process(x)
        }
    """

    variable: str
    iterable: Expression
    body: Block
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_for_statement(self)


@dataclass(frozen=True, slots=True)
class AsyncForStatement(Statement):
    """
    An async for loop statement for iterating over async iterators.

    Example:
        async for item in stream {
            println("Got: {}", item)
        }

        async for chunk in file.read_chunks() {
            process(chunk)
        }
    """

    variable: str
    iterable: Expression
    body: Block
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_async_for_statement(self)


@dataclass(frozen=True, slots=True)
class WhileStatement(Statement):
    """
    A while loop statement.

    Example:
        while x > 0 {
            x -= 1
        }
    """

    condition: Expression
    body: Block
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_while_statement(self)


@dataclass(frozen=True, slots=True)
class ReturnStatement(Statement):
    """
    A return statement.

    Example:
        return x + y
        return
    """

    value: Optional[Expression] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_return_statement(self)


@dataclass(frozen=True, slots=True)
class BreakStatement(Statement):
    """A break statement."""

    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_break_statement(self)


@dataclass(frozen=True, slots=True)
class ContinueStatement(Statement):
    """A continue statement."""

    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_continue_statement(self)


@dataclass(frozen=True, slots=True)
class PassStatement(Statement):
    """A pass statement (no-op)."""

    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_pass_statement(self)


@dataclass(frozen=True, slots=True)
class ImportStatement(Statement):
    """
    An import statement.

    Examples:
        import manim
        import numpy as np
        from manim import Circle, Square
    """

    module: str
    alias: Optional[str] = None
    names: tuple[tuple[str, Optional[str]], ...] = ()  # (name, alias) pairs
    is_from_import: bool = False
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_import_statement(self)


@dataclass(frozen=True, slots=True)
class PrintStatement(Statement):
    """
    A print statement (without newline).

    Example:
        print("Hello")
        print("Value: {}", x)
    """

    format_string: Expression
    arguments: tuple[Expression, ...] = ()
    newline: bool = False  # True for println
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_print_statement(self)


@dataclass(frozen=True, slots=True)
class UseStatement(Statement):
    """
    A use statement for importing modules/libraries.

    Examples:
        use manim.*           # Import all from manim
        use mylib.topology    # Import specific module
        use std.io            # Standard I/O
    """

    module_path: tuple[str, ...]
    wildcard: bool = False  # True for use module.*
    alias: Optional[str] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_use_statement(self)


@dataclass(frozen=True, slots=True)
class ModuleDecl(Statement):
    """
    A module declaration.

    Example:
        mod topology {
            pub fn euler_characteristic(v: Int, e: Int, f: Int) -> Int {
                return v - e + f
            }
        }
    """

    name: str
    body: Block
    is_public: bool = False
    doc_comment: Optional["DocComment"] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_module_decl(self)


@dataclass(frozen=True, slots=True)
class PlayStatement(Statement):
    """
    A Manim play statement for animations.

    Example:
        play(Create(circle))
        play(Transform(a, b), run_time=2.0)
    """

    animation: Expression
    run_time: Optional[Expression] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_play_statement(self)


@dataclass(frozen=True, slots=True)
class WaitStatement(Statement):
    """
    A Manim wait statement.

    Example:
        wait(1.0)
        wait()
    """

    duration: Optional[Expression] = None
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_wait_statement(self)


# -----------------------------------------------------------------------------
# OOP Constructs (Structs, Traits, Enums, Impl Blocks)
# -----------------------------------------------------------------------------


class Visibility(Enum):
    """Visibility modifier for struct fields and methods."""

    PRIVATE = auto()  # Default visibility
    PUBLIC = auto()   # pub keyword


@dataclass(frozen=True, slots=True)
class StructField:
    """
    A field definition in a struct.

    Example:
        x: Float
        pub name: String
    """

    name: str
    type_annotation: TypeAnnotation
    visibility: Visibility = Visibility.PRIVATE
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class StructDef(Statement):
    """
    A struct definition (lightweight data type) with optional generics.

    Example:
        struct Point {
            x: Float
            y: Float
        }

        struct Box<T> {
            value: T
        }

        struct Pair<A, B> {
            first: A
            second: B
        }
    """

    name: str
    fields: tuple[StructField, ...]
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    doc_comment: Optional["DocComment"] = None
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this struct has type parameters."""
        return len(self.type_params) > 0

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_struct_def(self)


@dataclass(frozen=True, slots=True)
class Method:
    """
    A method definition within an impl block, with optional generics.

    Methods have an optional 'self' parameter as first parameter.

    Example:
        fn distance(self, other: Point) -> Float { ... }
        fn origin() -> Point { ... }  # Static method (no self)
        fn transform<U>(self, f: (T) -> U) -> Box<U> { ... }  # Generic method
    """

    name: str
    parameters: tuple[Parameter, ...]  # First param may be 'self'
    return_type: Optional[TypeAnnotation] = None
    body: Optional["Block"] = None
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    visibility: Visibility = Visibility.PRIVATE
    has_self: bool = False  # True if first param is 'self'
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this method has its own type parameters."""
        return len(self.type_params) > 0


@dataclass(frozen=True, slots=True)
class AssociatedType:
    """
    An associated type declaration in a trait impl block.

    Example:
        impl Add for Vector {
            type Output = Vector  // This is an AssociatedType
            fn add(self, other: Vector) -> Vector { ... }
        }

    Attributes:
        name: The name of the associated type (e.g., "Output")
        type_value: The concrete type being assigned
        location: Source location for error reporting
    """

    name: str
    type_value: TypeAnnotation
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class ImplBlock(Statement):
    """
    An implementation block for a struct or trait, with optional generics.

    Example:
        impl Point {
            fn distance(self, other: Point) -> Float { ... }
        }

        impl Shape for Circle {
            fn area(self) -> Float { ... }
        }

        impl<T> Box<T> {
            fn new(value: T) -> Box<T> { ... }
            fn unwrap(self) -> T { ... }
        }

        impl<T: Display> Box<T> {
            fn print(self) { ... }
        }

        impl Add for Vector {
            type Output = Vector
            fn add(self, other: Vector) -> Vector { ... }
        }
    """

    target_type: str  # The struct/type being implemented
    trait_name: Optional[str] = None  # None for inherent impl, trait name for trait impl
    trait_type_args: tuple[TypeAnnotation, ...] = ()  # Type args for trait (e.g., Float in Mul<Float>)
    methods: tuple[Method, ...] = ()
    associated_types: tuple[AssociatedType, ...] = ()  # Associated type declarations
    type_params: tuple["TypeParameter", ...] = ()
    target_type_args: tuple[str, ...] = ()  # Type arguments for target (e.g., T in Box<T>)
    where_clause: Optional["WhereClause"] = None
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this impl block has type parameters."""
        return len(self.type_params) > 0

    @property
    def is_operator_impl(self) -> bool:
        """Check if this impl block is for an operator trait."""
        from mathviz.compiler.operators import OPERATOR_TRAITS
        return self.trait_name is not None and self.trait_name in OPERATOR_TRAITS

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_impl_block(self)


@dataclass(frozen=True, slots=True)
class TraitMethod:
    """
    A method signature in a trait definition, with optional generics.

    Example:
        fn area(self) -> Float
        fn perimeter(self) -> Float
        fn transform<U>(self, f: (T) -> U) -> U  // Generic method in trait
    """

    name: str
    parameters: tuple[Parameter, ...]
    return_type: Optional[TypeAnnotation] = None
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    has_self: bool = False
    has_default_impl: bool = False
    default_body: Optional["Block"] = None
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class TraitDef(Statement):
    """
    A trait (interface) definition with optional generics.

    Example:
        trait Shape {
            fn area(self) -> Float
            fn perimeter(self) -> Float
        }

        trait Container<T> {
            fn get(self) -> T
            fn set(self, value: T)
        }

        trait Comparable<T> {
            fn compare(self, other: T) -> Int
        }
    """

    name: str
    methods: tuple[TraitMethod, ...]
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    doc_comment: Optional["DocComment"] = None
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this trait has type parameters."""
        return len(self.type_params) > 0

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_trait_def(self)


@dataclass(frozen=True, slots=True)
class EnumVariant:
    """
    A variant of an enum type.

    Example:
        Circle(Float)           # With associated data
        Rectangle(Float, Float) # With multiple data
        Point                   # Without data
    """

    name: str
    fields: tuple[TypeAnnotation, ...] = ()  # Associated data types
    location: Optional[SourceLocation] = None


@dataclass(frozen=True, slots=True)
class EnumDef(Statement):
    """
    An enum (algebraic data type) definition with optional generics.

    Example:
        enum Shape {
            Circle(Float)
            Rectangle(Float, Float)
            Point
        }

        enum Option<T> {
            Some(T)
            None
        }

        enum Result<T, E> {
            Ok(T)
            Err(E)
        }
    """

    name: str
    variants: tuple[EnumVariant, ...]
    type_params: tuple["TypeParameter", ...] = ()
    where_clause: Optional["WhereClause"] = None
    doc_comment: Optional["DocComment"] = None
    location: Optional[SourceLocation] = None

    @property
    def is_generic(self) -> bool:
        """Check if this enum has type parameters."""
        return len(self.type_params) > 0

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_enum_def(self)


@dataclass(frozen=True, slots=True)
class SelfExpression(Expression):
    """
    Reference to 'self' within a method.

    Example:
        self.x
        self.radius
    """

    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_self_expression(self)


@dataclass(frozen=True, slots=True)
class EnumVariantAccess(Expression):
    """
    Access to an enum variant using :: syntax.

    Example:
        Shape::Circle(5.0)
        Shape::Point
        Color::Red
    """

    enum_name: str
    variant_name: str
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_enum_variant_access(self)


@dataclass(frozen=True, slots=True)
class StructLiteral(Expression):
    """
    A struct literal with named fields.

    Example:
        Point { x: 1.0, y: 2.0 }
    """

    struct_name: str
    fields: tuple[tuple[str, Expression], ...]  # (field_name, value) pairs
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_struct_literal(self)


@dataclass(frozen=True, slots=True)
class EnumPattern(Pattern):
    """
    Pattern matching an enum variant.

    Example:
        Shape::Circle(r)
        Shape::Rectangle(w, h)
        Shape::Point
    """

    enum_name: str
    variant_name: str
    bindings: tuple[Pattern, ...] = ()
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_enum_pattern(self)

    def get_bound_names(self) -> set[str]:
        names: set[str] = set()
        for binding in self.bindings:
            names.update(binding.get_bound_names())
        return names


# -----------------------------------------------------------------------------
# Program (Root Node)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Program(ASTNode):
    """
    The root node of a MathViz program.

    Contains all top-level statements.
    """

    statements: tuple[Statement, ...]
    location: Optional[SourceLocation] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_program(self)


# -----------------------------------------------------------------------------
# Visitor with default implementations
# -----------------------------------------------------------------------------


class BaseASTVisitor(ASTVisitor):
    """
    Base visitor with default implementations that traverse children.

    Subclass this and override specific visit_* methods as needed.
    """

    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            self.visit(stmt)

    def visit_block(self, node: Block) -> Any:
        for stmt in node.statements:
            self.visit(stmt)

    # Types
    def visit_simple_type(self, node: SimpleType) -> Any:
        pass

    def visit_generic_type(self, node: GenericType) -> Any:
        for arg in node.type_args:
            self.visit(arg)

    def visit_function_type(self, node: FunctionType) -> Any:
        for param in node.param_types:
            self.visit(param)
        self.visit(node.return_type)

    # Literals
    def visit_identifier(self, node: Identifier) -> Any:
        pass

    def visit_integer_literal(self, node: IntegerLiteral) -> Any:
        pass

    def visit_float_literal(self, node: FloatLiteral) -> Any:
        pass

    def visit_string_literal(self, node: StringLiteral) -> Any:
        pass

    def visit_fstring(self, node: "FString") -> Any:
        """Visit an f-string, traversing all expression parts."""
        for part in node.parts:
            if isinstance(part, FStringExpression):
                self.visit(part.expression)

    def visit_boolean_literal(self, node: BooleanLiteral) -> Any:
        pass

    def visit_none_literal(self, node: NoneLiteral) -> Any:
        pass

    def visit_list_literal(self, node: ListLiteral) -> Any:
        for elem in node.elements:
            self.visit(elem)

    def visit_set_literal(self, node: SetLiteral) -> Any:
        for elem in node.elements:
            self.visit(elem)

    def visit_dict_literal(self, node: DictLiteral) -> Any:
        for key, value in node.pairs:
            self.visit(key)
            self.visit(value)

    def visit_tuple_literal(self, node: TupleLiteral) -> Any:
        for elem in node.elements:
            self.visit(elem)

    def visit_some_expression(self, node: SomeExpression) -> Any:
        self.visit(node.value)

    def visit_ok_expression(self, node: OkExpression) -> Any:
        self.visit(node.value)

    def visit_err_expression(self, node: ErrExpression) -> Any:
        self.visit(node.value)

    def visit_unwrap_expression(self, node: UnwrapExpression) -> Any:
        self.visit(node.operand)

    def visit_await_expression(self, node: "AwaitExpression") -> Any:
        """Visit an await expression."""
        self.visit(node.expression)

    # Expressions
    def visit_binary_expression(self, node: BinaryExpression) -> Any:
        self.visit(node.left)
        self.visit(node.right)

    def visit_unary_expression(self, node: UnaryExpression) -> Any:
        self.visit(node.operand)

    def visit_call_expression(self, node: CallExpression) -> Any:
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)

    def visit_member_access(self, node: MemberAccess) -> Any:
        self.visit(node.object)

    def visit_index_expression(self, node: IndexExpression) -> Any:
        self.visit(node.object)
        self.visit(node.index)

    def visit_conditional_expression(self, node: ConditionalExpression) -> Any:
        self.visit(node.condition)
        self.visit(node.then_expr)
        self.visit(node.else_expr)

    def visit_lambda_expression(self, node: LambdaExpression) -> Any:
        self.visit(node.body)

    def visit_range_expression(self, node: RangeExpression) -> Any:
        self.visit(node.start)
        self.visit(node.end)
        if node.step:
            self.visit(node.step)

    # Comprehensions
    def visit_list_comprehension(self, node: "ListComprehension") -> Any:
        self.visit(node.element)
        for clause in node.clauses:
            self.visit(clause.iterable)
            if clause.condition:
                self.visit(clause.condition)

    def visit_set_comprehension(self, node: "SetComprehension") -> Any:
        self.visit(node.element)
        for clause in node.clauses:
            self.visit(clause.iterable)
            if clause.condition:
                self.visit(clause.condition)

    def visit_dict_comprehension(self, node: "DictComprehension") -> Any:
        self.visit(node.key)
        self.visit(node.value)
        for clause in node.clauses:
            self.visit(clause.iterable)
            if clause.condition:
                self.visit(clause.condition)

    def visit_pipe_lambda(self, node: "PipeLambda") -> Any:
        self.visit(node.body)

    # Pattern Matching
    def visit_literal_pattern(self, node: LiteralPattern) -> Any:
        self.visit(node.value)

    def visit_identifier_pattern(self, node: IdentifierPattern) -> Any:
        pass

    def visit_tuple_pattern(self, node: TuplePattern) -> Any:
        for elem in node.elements:
            elem.accept(self)

    def visit_constructor_pattern(self, node: ConstructorPattern) -> Any:
        for arg in node.args:
            arg.accept(self)

    def visit_range_pattern(self, node: "RangePattern") -> Any:
        """Visit a range pattern, traversing start and end expressions."""
        self.visit(node.start)
        self.visit(node.end)

    def visit_or_pattern(self, node: "OrPattern") -> Any:
        """Visit an or pattern, traversing all alternative patterns."""
        for pattern in node.patterns:
            pattern.accept(self)

    def visit_binding_pattern(self, node: "BindingPattern") -> Any:
        """Visit a binding pattern, traversing the inner pattern."""
        node.pattern.accept(self)

    def visit_rest_pattern(self, node: "RestPattern") -> Any:
        """Visit a rest pattern (no children to traverse)."""
        pass

    def visit_list_pattern(self, node: "ListPattern") -> Any:
        """Visit a list pattern, traversing all element patterns."""
        for elem in node.elements:
            elem.accept(self)

    def visit_match_expression(self, node: MatchExpression) -> Any:
        self.visit(node.subject)
        for arm in node.arms:
            arm.pattern.accept(self)
            if arm.guard:
                self.visit(arm.guard)
            if isinstance(arm.body, Block):
                self.visit(arm.body)
            else:
                self.visit(arm.body)

    # Statements
    def visit_expression_statement(self, node: ExpressionStatement) -> Any:
        self.visit(node.expression)

    def visit_let_statement(self, node: LetStatement) -> Any:
        if node.type_annotation:
            self.visit(node.type_annotation)
        if node.value:
            self.visit(node.value)

    def visit_const_declaration(self, node: "ConstDeclaration") -> Any:
        """Visit a const declaration."""
        if node.type_annotation:
            self.visit(node.type_annotation)
        self.visit(node.value)

    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        self.visit(node.target)
        self.visit(node.value)

    def visit_compound_assignment(self, node: CompoundAssignment) -> Any:
        self.visit(node.target)
        self.visit(node.value)

    def visit_function_def(self, node: FunctionDef) -> Any:
        for param in node.parameters:
            if param.type_annotation:
                self.visit(param.type_annotation)
            if param.default_value:
                self.visit(param.default_value)
        if node.return_type:
            self.visit(node.return_type)
        if node.body:
            self.visit(node.body)

    def visit_async_function_def(self, node: "AsyncFunctionDef") -> Any:
        """Visit an async function definition."""
        for param in node.parameters:
            if param.type_annotation:
                self.visit(param.type_annotation)
            if param.default_value:
                self.visit(param.default_value)
        if node.return_type:
            self.visit(node.return_type)
        if node.body:
            self.visit(node.body)

    def visit_class_def(self, node: ClassDef) -> Any:
        if node.body:
            self.visit(node.body)

    def visit_scene_def(self, node: SceneDef) -> Any:
        if node.body:
            self.visit(node.body)

    def visit_if_statement(self, node: IfStatement) -> Any:
        self.visit(node.condition)
        self.visit(node.then_block)
        for condition, block in node.elif_clauses:
            self.visit(condition)
            self.visit(block)
        if node.else_block:
            self.visit(node.else_block)

    def visit_for_statement(self, node: ForStatement) -> Any:
        self.visit(node.iterable)
        self.visit(node.body)

    def visit_async_for_statement(self, node: "AsyncForStatement") -> Any:
        """Visit an async for statement."""
        self.visit(node.iterable)
        self.visit(node.body)

    def visit_while_statement(self, node: WhileStatement) -> Any:
        self.visit(node.condition)
        self.visit(node.body)

    def visit_return_statement(self, node: ReturnStatement) -> Any:
        if node.value:
            self.visit(node.value)

    def visit_break_statement(self, node: BreakStatement) -> Any:
        pass

    def visit_continue_statement(self, node: ContinueStatement) -> Any:
        pass

    def visit_pass_statement(self, node: PassStatement) -> Any:
        pass

    def visit_import_statement(self, node: ImportStatement) -> Any:
        pass

    def visit_print_statement(self, node: PrintStatement) -> Any:
        self.visit(node.format_string)
        for arg in node.arguments:
            self.visit(arg)

    def visit_use_statement(self, node: UseStatement) -> Any:
        pass

    def visit_module_decl(self, node: ModuleDecl) -> Any:
        self.visit(node.body)

    def visit_play_statement(self, node: PlayStatement) -> Any:
        self.visit(node.animation)
        if node.run_time:
            self.visit(node.run_time)

    def visit_wait_statement(self, node: WaitStatement) -> Any:
        if node.duration:
            self.visit(node.duration)

    # OOP Constructs
    def visit_struct_def(self, node: "StructDef") -> Any:
        for field in node.fields:
            self.visit(field.type_annotation)

    def visit_impl_block(self, node: "ImplBlock") -> Any:
        for method in node.methods:
            for param in method.parameters:
                if param.type_annotation:
                    self.visit(param.type_annotation)
                if param.default_value:
                    self.visit(param.default_value)
            if method.return_type:
                self.visit(method.return_type)
            if method.body:
                self.visit(method.body)

    def visit_trait_def(self, node: "TraitDef") -> Any:
        for method in node.methods:
            for param in method.parameters:
                if param.type_annotation:
                    self.visit(param.type_annotation)
            if method.return_type:
                self.visit(method.return_type)
            if method.default_body:
                self.visit(method.default_body)

    def visit_enum_def(self, node: "EnumDef") -> Any:
        for variant in node.variants:
            for field in variant.fields:
                self.visit(field)

    def visit_self_expression(self, node: "SelfExpression") -> Any:
        pass

    def visit_enum_variant_access(self, node: "EnumVariantAccess") -> Any:
        pass

    def visit_struct_literal(self, node: "StructLiteral") -> Any:
        for _, value in node.fields:
            self.visit(value)

    def visit_enum_pattern(self, node: "EnumPattern") -> Any:
        for binding in node.bindings:
            binding.accept(self)
