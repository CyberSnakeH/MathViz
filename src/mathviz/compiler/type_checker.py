"""
Type Inference and Checking Module for MathViz Compiler.

This module implements a complete type inference and checking system that:
1. Infers types for all expressions (literals, binary ops, function calls, etc.)
2. Propagates types through the data flow graph
3. Checks type consistency (e.g., `let x: Int = "hello"` should error)
4. Detects implicit conversions (Int to Float, etc.)
5. Reports type errors with source location

The type system supports:
- Primitive types: Int, Float, Bool, String, None
- Collection types: List[T], Set[T], Dict[K,V]
- Mathematical types: Vec, Mat, Array
- Function types: (T1, T2, ...) -> R
- Type inference and unification
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union, Sequence

from mathviz.compiler.ast_nodes import (
    # Program and base
    Program,
    ASTNode,
    BaseASTVisitor,
    Block,
    # Type annotations
    TypeAnnotation,
    SimpleType,
    GenericType,
    FunctionType as ASTFunctionType,
    # Expressions
    Expression,
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BooleanLiteral,
    NoneLiteral,
    ListLiteral,
    SetLiteral,
    DictLiteral,
    TupleLiteral,
    SomeExpression,
    OkExpression,
    ErrExpression,
    UnwrapExpression,
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    MemberAccess,
    IndexExpression,
    ConditionalExpression,
    LambdaExpression,
    RangeExpression,
    BinaryOperator,
    UnaryOperator,
    # Pattern matching
    Pattern,
    LiteralPattern,
    IdentifierPattern,
    TuplePattern,
    ConstructorPattern,
    RangePattern,
    OrPattern,
    BindingPattern,
    RestPattern,
    ListPattern,
    MatchArm,
    MatchExpression,
    # F-strings
    FStringPart,
    FStringLiteral,
    FStringExpression,
    FString,
    # Statements
    Statement,
    ExpressionStatement,
    LetStatement,
    ConstDeclaration,
    AssignmentStatement,
    CompoundAssignment,
    FunctionDef,
    ClassDef,
    SceneDef,
    IfStatement,
    ForStatement,
    WhileStatement,
    ReturnStatement,
    BreakStatement,
    ContinueStatement,
    PassStatement,
    ImportStatement,
    PrintStatement,
    UseStatement,
    ModuleDecl,
    PlayStatement,
    WaitStatement,
    Parameter,
    # OOP constructs
    Visibility,
    StructField,
    StructDef,
    Method,
    ImplBlock,
    AssociatedType,
    TraitMethod,
    TraitDef,
    EnumVariant,
    EnumDef,
    SelfExpression,
    EnumVariantAccess,
    StructLiteral,
    EnumPattern,
)
from mathviz.compiler.operators import (
    OPERATOR_TRAITS,
    OperatorKind,
    get_trait_for_binary_operator,
    get_trait_for_unary_operator,
    is_operator_trait,
)
from mathviz.utils.errors import SourceLocation, TypeError as MathVizTypeError
from mathviz.utils.diagnostics import (
    DiagnosticEmitter,
    DiagnosticLevel,
    SourceSpan,
    Diagnostic,
    ErrorCode,
    VariableInfo,
    suggest_similar,
    create_undefined_variable_diagnostic,
    create_type_mismatch_diagnostic,
    create_undefined_function_diagnostic,
    create_wrong_arguments_diagnostic,
    create_break_outside_loop_diagnostic,
    create_return_outside_function_diagnostic,
)


# =============================================================================
# Type System Representation
# =============================================================================


class Type(ABC):
    """
    Base class for all types in the MathViz type system.

    Types are immutable and support structural equality.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the type."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check structural equality between types."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Return hash for use in dictionaries and sets."""
        pass

    def is_numeric(self) -> bool:
        """Check if this type is numeric (Int or Float)."""
        return False

    def is_compatible_with(self, other: Type) -> bool:
        """
        Check if this type can be used where 'other' is expected.

        This handles subtyping relationships and implicit conversions.
        """
        if self == other:
            return True
        if isinstance(other, UnknownType) or isinstance(self, UnknownType):
            return True
        if isinstance(other, AnyType) or isinstance(self, AnyType):
            return True
        return False


@dataclass(frozen=True)
class PrimitiveType(Type):
    """
    A primitive (built-in) type.

    Supported primitives: Int, Float, Bool, String, None
    """

    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrimitiveType):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("primitive", self.name))

    def is_numeric(self) -> bool:
        return self.name in ("Int", "Float")

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        # Int is compatible with Float (implicit widening conversion)
        if self.name == "Int" and isinstance(other, PrimitiveType) and other.name == "Float":
            return True
        return False


# Singleton instances for primitive types
INT_TYPE = PrimitiveType("Int")
FLOAT_TYPE = PrimitiveType("Float")
BOOL_TYPE = PrimitiveType("Bool")
STRING_TYPE = PrimitiveType("String")
NONE_TYPE = PrimitiveType("None")


@dataclass(frozen=True)
class GenericTypeInstance(Type):
    """
    A generic type instantiated with specific type arguments.

    Examples: List[Int], Set[String], Dict[String, Int]
    """

    base_name: str
    type_args: tuple[Type, ...]

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.type_args)
        return f"{self.base_name}[{args_str}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenericTypeInstance):
            return False
        return self.base_name == other.base_name and self.type_args == other.type_args

    def __hash__(self) -> int:
        return hash(("generic", self.base_name, self.type_args))

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        if isinstance(other, GenericTypeInstance):
            if self.base_name != other.base_name:
                return False
            if len(self.type_args) != len(other.type_args):
                return False
            # Covariant check for element types
            return all(
                self_arg.is_compatible_with(other_arg)
                for self_arg, other_arg in zip(self.type_args, other.type_args)
            )
        return False


@dataclass(frozen=True)
class ArrayType(Type):
    """
    An array type representing mathematical arrays/vectors/matrices.

    Examples: Array, Vec, Mat, Array[Float]
    """

    name: str  # "Array", "Vec", or "Mat"
    element_type: Optional[Type] = None
    dimensions: Optional[tuple[int, ...]] = None

    def __str__(self) -> str:
        if self.element_type is None:
            if self.dimensions:
                return f"{self.name}{list(self.dimensions)}"
            return self.name
        if self.dimensions:
            return f"{self.name}[{self.element_type}]{list(self.dimensions)}"
        return f"{self.name}[{self.element_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayType):
            return False
        return (
            self.name == other.name
            and self.element_type == other.element_type
            and self.dimensions == other.dimensions
        )

    def __hash__(self) -> int:
        return hash(("array", self.name, self.element_type, self.dimensions))

    def is_numeric(self) -> bool:
        # Arrays of numerics are considered numeric for arithmetic operations
        if self.element_type is None:
            return True  # Assume numeric by default
        return self.element_type.is_numeric()

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        if isinstance(other, ArrayType):
            # Vec and Mat are subtypes of Array
            if other.name == "Array" and self.name in ("Vec", "Mat"):
                if other.element_type is None:
                    return True
                if self.element_type and self.element_type.is_compatible_with(other.element_type):
                    return True
            # Same array type with compatible element type
            if self.name == other.name:
                if other.element_type is None:
                    return True
                if self.element_type and self.element_type.is_compatible_with(other.element_type):
                    return True
        return False


# Singleton instances for mathematical array types
VEC_TYPE = ArrayType("Vec", FLOAT_TYPE)
MAT_TYPE = ArrayType("Mat", FLOAT_TYPE)
ARRAY_TYPE = ArrayType("Array")


@dataclass(frozen=True)
class FunctionType(Type):
    """
    A function type representing callable signatures.

    Example: (Int, Int) -> Int
    """

    param_types: tuple[Type, ...]
    return_type: Type
    param_names: tuple[str, ...] = ()  # Optional parameter names

    def __str__(self) -> str:
        if not self.param_types:
            return f"() -> {self.return_type}"
        params_str = ", ".join(str(t) for t in self.param_types)
        return f"({params_str}) -> {self.return_type}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionType):
            return False
        return (
            self.param_types == other.param_types
            and self.return_type == other.return_type
        )

    def __hash__(self) -> int:
        return hash(("function", self.param_types, self.return_type))

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        if isinstance(other, FunctionType):
            # Contravariant in parameters, covariant in return type
            if len(self.param_types) != len(other.param_types):
                return False
            # Parameters are contravariant
            for other_param, self_param in zip(other.param_types, self.param_types):
                if not other_param.is_compatible_with(self_param):
                    return False
            # Return type is covariant
            return self.return_type.is_compatible_with(other.return_type)
        return False


@dataclass(frozen=True)
class ClassType(Type):
    """
    A class/user-defined type.

    Example: Point, Circle, MyScene
    """

    name: str
    base_classes: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassType):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("class", self.name))

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        if isinstance(other, ClassType):
            # Check if we inherit from the other class
            return other.name in self.base_classes
        return False


@dataclass(frozen=True)
class StructType(Type):
    """
    A struct type representing a lightweight data structure.

    Example: Point, Rectangle, Person
    """

    name: str
    fields: tuple[tuple[str, Type], ...] = ()  # (field_name, field_type) pairs

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructType):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("struct", self.name))

    def get_field_type(self, field_name: str) -> Optional[Type]:
        """Get the type of a field by name."""
        for name, type_ in self.fields:
            if name == field_name:
                return type_
        return None


@dataclass(frozen=True)
class TraitType(Type):
    """
    A trait type representing an interface.

    Example: Shape, Drawable, Iterator
    """

    name: str
    methods: tuple[tuple[str, FunctionType], ...] = ()  # (method_name, signature) pairs

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TraitType):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("trait", self.name))


@dataclass(frozen=True)
class EnumType(Type):
    """
    An enum type representing a set of variants.

    Example: Color, Shape, Result
    """

    name: str
    variants: tuple[tuple[str, tuple[Type, ...]], ...] = ()  # (variant_name, associated_types) pairs

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnumType):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("enum", self.name))

    def get_variant(self, variant_name: str) -> Optional[tuple[Type, ...]]:
        """Get the associated types for a variant by name."""
        for name, types in self.variants:
            if name == variant_name:
                return types
        return None


@dataclass(frozen=True)
class EnumVariantType(Type):
    """
    The type of an enum variant constructor.

    Example: Shape.Circle, Color.Red
    """

    enum_type: EnumType
    variant_name: str
    associated_types: tuple[Type, ...] = ()

    def __str__(self) -> str:
        if self.associated_types:
            types_str = ", ".join(str(t) for t in self.associated_types)
            return f"{self.enum_type.name}::{self.variant_name}({types_str})"
        return f"{self.enum_type.name}::{self.variant_name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnumVariantType):
            return False
        return (
            self.enum_type == other.enum_type
            and self.variant_name == other.variant_name
        )

    def __hash__(self) -> int:
        return hash(("enum_variant", self.enum_type.name, self.variant_name))


@dataclass(frozen=True)
class UnknownType(Type):
    """
    Represents an unknown or not-yet-inferred type.

    Used during type inference for unresolved types.
    """

    id: int = 0  # Unique identifier for this unknown type

    def __str__(self) -> str:
        return f"?T{self.id}" if self.id > 0 else "?"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnknownType):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(("unknown", self.id))

    def is_compatible_with(self, other: Type) -> bool:
        # Unknown is compatible with everything (will be unified later)
        return True


@dataclass(frozen=True)
class AnyType(Type):
    """
    The top type that is compatible with everything.

    Used for dynamic typing contexts and external library calls.
    """

    def __str__(self) -> str:
        return "Any"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnyType)

    def __hash__(self) -> int:
        return hash("any")

    def is_compatible_with(self, other: Type) -> bool:
        return True


# Singleton instances
UNKNOWN_TYPE = UnknownType(0)
ANY_TYPE = AnyType()


@dataclass(frozen=True)
class RangeType(Type):
    """
    The type of range expressions (0..10, 1..=100).

    Ranges are iterable and produce integers.
    """

    element_type: Type = field(default_factory=lambda: INT_TYPE)

    def __str__(self) -> str:
        return f"Range[{self.element_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RangeType):
            return False
        return self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash(("range", self.element_type))


RANGE_TYPE = RangeType(INT_TYPE)


@dataclass(frozen=True)
class TupleType(Type):
    """
    A tuple type representing fixed-size heterogeneous sequences.

    Examples: (Int, String), (Float, Float, Float), ()
    """

    element_types: tuple[Type, ...]

    def __str__(self) -> str:
        if not self.element_types:
            return "()"
        types_str = ", ".join(str(t) for t in self.element_types)
        if len(self.element_types) == 1:
            return f"({types_str},)"
        return f"({types_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleType):
            return False
        return self.element_types == other.element_types

    def __hash__(self) -> int:
        return hash(("tuple", self.element_types))

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        if isinstance(other, TupleType):
            if len(self.element_types) != len(other.element_types):
                return False
            return all(
                self_elem.is_compatible_with(other_elem)
                for self_elem, other_elem in zip(self.element_types, other.element_types)
            )
        return False


# Empty tuple type singleton
EMPTY_TUPLE_TYPE = TupleType(())


@dataclass(frozen=True)
class OptionalType(Type):
    """
    An optional type representing a value that may or may not be present.

    Example: Optional[Int] = Int | None

    Values can be constructed with:
    - Some(value) - a present value
    - None - an absent value
    """

    inner_type: Type

    def __str__(self) -> str:
        return f"Optional[{self.inner_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionalType):
            return False
        return self.inner_type == other.inner_type

    def __hash__(self) -> int:
        return hash(("optional", self.inner_type))

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        # Optional[T] is compatible with Optional[T]
        if isinstance(other, OptionalType):
            return self.inner_type.is_compatible_with(other.inner_type)
        # Optional[T] accepts None
        if other == NONE_TYPE:
            return True
        # Optional[T] is compatible with T (unwrapped value assignment)
        return self.inner_type.is_compatible_with(other)

    def unwrap_type(self) -> Type:
        """Get the inner type when unwrapped."""
        return self.inner_type


@dataclass(frozen=True)
class ResultType(Type):
    """
    A result type for error handling, representing either success or failure.

    Example: Result[Int, String] = Ok(Int) | Err(String)

    Values can be constructed with:
    - Ok(value) - a successful result
    - Err(error) - an error result
    """

    ok_type: Type
    err_type: Type

    def __str__(self) -> str:
        return f"Result[{self.ok_type}, {self.err_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultType):
            return False
        return self.ok_type == other.ok_type and self.err_type == other.err_type

    def __hash__(self) -> int:
        return hash(("result", self.ok_type, self.err_type))

    def is_compatible_with(self, other: Type) -> bool:
        if super().is_compatible_with(other):
            return True
        if isinstance(other, ResultType):
            return (
                self.ok_type.is_compatible_with(other.ok_type)
                and self.err_type.is_compatible_with(other.err_type)
            )
        return False

    def unwrap_type(self) -> Type:
        """Get the Ok type when unwrapped."""
        return self.ok_type

    def error_type(self) -> Type:
        """Get the Err type."""
        return self.err_type


# =============================================================================
# Generic Type Variables
# =============================================================================


class TypeVariable(Type):
    """
    A type variable representing a generic type parameter.

    Used in generic functions, structs, enums, and traits.
    Type variables are bound to concrete types during instantiation.

    Examples:
        T in fn identity<T>(x: T) -> T
        A, B in struct Pair<A, B>
        T in enum Option<T>
    """

    def __init__(self, name: str, bounds: tuple[str, ...] = (), id: int = 0) -> None:
        """
        Initialize a type variable.

        Args:
            name: The name of the type variable (e.g., "T", "U")
            bounds: Trait bounds the type must satisfy (e.g., ("Display", "Clone"))
            id: Unique identifier for distinguishing type variables with the same name
        """
        self.name = name
        self.bounds = bounds
        self.id = id

    def __str__(self) -> str:
        if self.bounds:
            return f"{self.name}: {' + '.join(self.bounds)}"
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeVariable):
            return self.name == other.name and self.id == other.id
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.id))

    def is_compatible_with(self, other: Type) -> bool:
        """Check if this type variable is compatible with another type."""
        # Type variables are compatible with anything (they get instantiated)
        if isinstance(other, (UnknownType, AnyType)):
            return True
        # Same type variable is compatible
        if isinstance(other, TypeVariable):
            return self.name == other.name and self.id == other.id
        # Type variables accept any concrete type (bounds checked separately)
        return True

    def satisfies_bounds(self, concrete_type: Type, trait_registry: dict[str, set[str]]) -> bool:
        """
        Check if a concrete type satisfies this type variable's bounds.

        Args:
            concrete_type: The type being substituted for this variable
            trait_registry: Map of type names to sets of implemented traits

        Returns:
            True if all bounds are satisfied
        """
        if not self.bounds:
            return True

        type_name = str(concrete_type)
        implemented_traits = trait_registry.get(type_name, set())

        for bound in self.bounds:
            if bound not in implemented_traits:
                return False
        return True


class GenericFunctionType(Type):
    """
    A generic function type with type parameters.

    Represents functions like:
        fn identity<T>(x: T) -> T
        fn map<T, U>(list: List[T], f: (T) -> U) -> List[U]
    """

    def __init__(
        self,
        type_params: tuple[TypeVariable, ...],
        param_types: tuple[Type, ...],
        return_type: Type,
        param_names: tuple[str, ...] = (),
    ) -> None:
        self.type_params = type_params
        self.param_types = param_types
        self.return_type = return_type
        self.param_names = param_names

    def __str__(self) -> str:
        type_params_str = ", ".join(str(tp) for tp in self.type_params)
        params_str = ", ".join(str(pt) for pt in self.param_types)
        return f"<{type_params_str}>({params_str}) -> {self.return_type}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericFunctionType):
            return (
                self.type_params == other.type_params
                and self.param_types == other.param_types
                and self.return_type == other.return_type
            )
        return False

    def __hash__(self) -> int:
        return hash((self.type_params, self.param_types, self.return_type))

    def is_compatible_with(self, other: Type) -> bool:
        """Generic functions need instantiation to be compared."""
        if isinstance(other, (UnknownType, AnyType)):
            return True
        if isinstance(other, GenericFunctionType):
            return self == other
        return False

    def instantiate(self, type_args: dict[str, Type]) -> FunctionType:
        """
        Instantiate this generic function with concrete type arguments.

        Args:
            type_args: Map of type parameter names to concrete types

        Returns:
            A concrete FunctionType with type variables substituted
        """
        def substitute(t: Type) -> Type:
            if isinstance(t, TypeVariable):
                return type_args.get(t.name, t)
            elif isinstance(t, GenericTypeInstance):
                new_args = tuple(substitute(arg) for arg in t.type_args)
                return GenericTypeInstance(t.base_name, new_args)
            elif isinstance(t, FunctionType):
                new_params = tuple(substitute(p) for p in t.param_types)
                new_return = substitute(t.return_type)
                return FunctionType(new_params, new_return, t.param_names)
            elif isinstance(t, TupleType):
                new_elements = tuple(substitute(e) for e in t.element_types)
                return TupleType(new_elements)
            elif isinstance(t, OptionalType):
                return OptionalType(substitute(t.inner_type))
            elif isinstance(t, ResultType):
                return ResultType(substitute(t.ok_type), substitute(t.err_type))
            return t

        new_params = tuple(substitute(pt) for pt in self.param_types)
        new_return = substitute(self.return_type)
        return FunctionType(new_params, new_return, self.param_names)


class GenericStructType(Type):
    """
    A generic struct type with type parameters.

    Represents structs like:
        struct Box<T> { value: T }
        struct Pair<A, B> { first: A, second: B }
    """

    def __init__(
        self,
        name: str,
        type_params: tuple[TypeVariable, ...],
        fields: dict[str, Type],
    ) -> None:
        self.name = name
        self.type_params = type_params
        self.fields = fields

    def __str__(self) -> str:
        type_params_str = ", ".join(tp.name for tp in self.type_params)
        return f"{self.name}<{type_params_str}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericStructType):
            return self.name == other.name and self.type_params == other.type_params
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.type_params))

    def instantiate(self, type_args: dict[str, Type]) -> StructType:
        """
        Instantiate this generic struct with concrete type arguments.

        Args:
            type_args: Map of type parameter names to concrete types

        Returns:
            A concrete StructType with type variables substituted
        """
        def substitute(t: Type) -> Type:
            if isinstance(t, TypeVariable):
                return type_args.get(t.name, t)
            elif isinstance(t, GenericTypeInstance):
                new_args = tuple(substitute(arg) for arg in t.type_args)
                return GenericTypeInstance(t.base_name, new_args)
            return t

        new_fields = {name: substitute(typ) for name, typ in self.fields.items()}

        # Create a name that includes the type arguments
        type_args_str = ", ".join(str(type_args.get(tp.name, tp.name)) for tp in self.type_params)
        instantiated_name = f"{self.name}[{type_args_str}]"

        return StructType(instantiated_name, new_fields)


class GenericEnumType(Type):
    """
    A generic enum type with type parameters.

    Represents enums like:
        enum Option<T> { Some(T), None }
        enum Result<T, E> { Ok(T), Err(E) }
    """

    def __init__(
        self,
        name: str,
        type_params: tuple[TypeVariable, ...],
        variants: dict[str, tuple[Type, ...]],
    ) -> None:
        self.name = name
        self.type_params = type_params
        self.variants = variants

    def __str__(self) -> str:
        type_params_str = ", ".join(tp.name for tp in self.type_params)
        return f"{self.name}<{type_params_str}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericEnumType):
            return self.name == other.name and self.type_params == other.type_params
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.type_params))

    def instantiate(self, type_args: dict[str, Type]) -> EnumType:
        """
        Instantiate this generic enum with concrete type arguments.

        Args:
            type_args: Map of type parameter names to concrete types

        Returns:
            A concrete EnumType with type variables substituted
        """
        def substitute(t: Type) -> Type:
            if isinstance(t, TypeVariable):
                return type_args.get(t.name, t)
            elif isinstance(t, GenericTypeInstance):
                new_args = tuple(substitute(arg) for arg in t.type_args)
                return GenericTypeInstance(t.base_name, new_args)
            return t

        new_variants = {}
        for name, field_types in self.variants.items():
            new_variants[name] = tuple(substitute(t) for t in field_types)

        # Create a name that includes the type arguments
        type_args_str = ", ".join(str(type_args.get(tp.name, tp.name)) for tp in self.type_params)
        instantiated_name = f"{self.name}[{type_args_str}]"

        return EnumType(instantiated_name, new_variants)


# =============================================================================
# Function Signature Registry
# =============================================================================


@dataclass
class FunctionSignature:
    """
    Stores complete function signature information.

    This includes parameter types with names, return type, and default values.
    """

    name: str
    param_types: list[Type]
    param_names: list[str]
    return_type: Type
    has_defaults: list[bool] = field(default_factory=list)
    location: Optional[SourceLocation] = None

    def to_function_type(self) -> FunctionType:
        """Convert to a FunctionType for type checking."""
        return FunctionType(
            param_types=tuple(self.param_types),
            return_type=self.return_type,
            param_names=tuple(self.param_names),
        )

    @property
    def min_args(self) -> int:
        """Minimum number of required arguments."""
        if not self.has_defaults:
            return len(self.param_types)
        return sum(1 for has_default in self.has_defaults if not has_default)

    @property
    def max_args(self) -> int:
        """Maximum number of arguments."""
        return len(self.param_types)


# =============================================================================
# Type Conversion and Compatibility
# =============================================================================


class TypeConversion(Enum):
    """Types of implicit conversions supported."""

    NONE = auto()           # No conversion needed
    INT_TO_FLOAT = auto()   # Widening conversion
    SUBTYPE = auto()        # Subtype relationship
    ANY = auto()            # Conversion to Any type


@dataclass(frozen=True)
class ConversionInfo:
    """Information about a type conversion."""

    from_type: Type
    to_type: Type
    conversion: TypeConversion
    location: Optional[SourceLocation] = None


# =============================================================================
# Symbol Table and Scope Management
# =============================================================================


@dataclass
class SymbolInfo:
    """
    Complete information about a symbol in a scope.

    Tracks the type and definition location for rich error messages.
    """
    type_: Type
    location: Optional[SourceLocation] = None
    is_parameter: bool = False
    is_mutable: bool = True


@dataclass
class Scope:
    """
    A single scope level containing variable bindings.

    Scopes form a chain from inner to outer for lexical scoping.
    """

    symbols: dict[str, SymbolInfo] = field(default_factory=dict)
    parent: Optional["Scope"] = None
    name: str = ""  # For debugging (e.g., "function:add", "block", "global")

    def lookup(self, name: str) -> Optional[Type]:
        """Look up a symbol type in this scope or parent scopes."""
        info = self.lookup_info(name)
        return info.type_ if info else None

    def lookup_info(self, name: str) -> Optional[SymbolInfo]:
        """Look up complete symbol info in this scope or parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup_info(name)
        return None

    def define(self, name: str, type_: Type, location: Optional[SourceLocation] = None,
               is_parameter: bool = False, is_mutable: bool = True) -> None:
        """Define a symbol in this scope with optional location tracking."""
        self.symbols[name] = SymbolInfo(type_, location, is_parameter, is_mutable)

    def is_defined_locally(self, name: str) -> bool:
        """Check if a symbol is defined in this immediate scope."""
        return name in self.symbols

    def get_all_names(self) -> list[str]:
        """Get all symbol names visible from this scope."""
        names = list(self.symbols.keys())
        if self.parent:
            names.extend(self.parent.get_all_names())
        return names


class SymbolTable:
    """
    Manages nested scopes for variable type tracking.

    Provides methods for entering/exiting scopes and looking up/defining symbols.
    """

    def __init__(self) -> None:
        self._global_scope = Scope(name="global")
        self._current_scope = self._global_scope

    @property
    def current_scope(self) -> Scope:
        """Get the current scope."""
        return self._current_scope

    def enter_scope(self, name: str = "") -> None:
        """Enter a new nested scope."""
        new_scope = Scope(parent=self._current_scope, name=name)
        self._current_scope = new_scope

    def exit_scope(self) -> None:
        """Exit the current scope, returning to the parent."""
        if self._current_scope.parent:
            self._current_scope = self._current_scope.parent
        # Don't exit global scope

    def lookup(self, name: str) -> Optional[Type]:
        """Look up a symbol in the current scope chain."""
        return self._current_scope.lookup(name)

    def lookup_info(self, name: str) -> Optional[SymbolInfo]:
        """Look up complete symbol info in the current scope chain."""
        return self._current_scope.lookup_info(name)

    def define(self, name: str, type_: Type, location: Optional[SourceLocation] = None,
               is_parameter: bool = False, is_mutable: bool = True) -> None:
        """Define a symbol in the current scope with optional location tracking."""
        self._current_scope.define(name, type_, location, is_parameter, is_mutable)

    def is_defined_locally(self, name: str) -> bool:
        """Check if a symbol is defined in the current scope."""
        return self._current_scope.is_defined_locally(name)

    def get_all_symbols(self) -> dict[str, Type]:
        """Get all symbols visible from the current scope."""
        result: dict[str, Type] = {}
        scope: Optional[Scope] = self._current_scope
        while scope:
            for name, info in scope.symbols.items():
                if name not in result:  # Inner scopes shadow outer
                    result[name] = info.type_
            scope = scope.parent
        return result

    def get_all_symbol_names(self) -> list[str]:
        """Get all symbol names visible from the current scope."""
        return self._current_scope.get_all_names()

    def get_variable_info(self, name: str) -> Optional[VariableInfo]:
        """Get VariableInfo for a symbol for diagnostics."""
        info = self.lookup_info(name)
        if info and info.location:
            span = SourceSpan.from_location(
                info.location.line,
                info.location.column,
                len(name),
                info.location.filename or "<input>",
            )
            return VariableInfo(
                name=name,
                type_name=str(info.type_),
                defined_at=span,
                is_mutable=info.is_mutable,
                is_parameter=info.is_parameter,
            )
        return None

    def get_all_variable_info(self) -> dict[str, VariableInfo]:
        """Get VariableInfo for all visible symbols."""
        result: dict[str, VariableInfo] = {}
        scope: Optional[Scope] = self._current_scope
        while scope:
            for name, info in scope.symbols.items():
                if name not in result and info.location:
                    span = SourceSpan.from_location(
                        info.location.line,
                        info.location.column,
                        len(name),
                        info.location.filename or "<input>",
                    )
                    result[name] = VariableInfo(
                        name=name,
                        type_name=str(info.type_),
                        defined_at=span,
                        is_mutable=info.is_mutable,
                        is_parameter=info.is_parameter,
                    )
            scope = scope.parent
        return result


# =============================================================================
# Type Checker Implementation
# =============================================================================


class TypeChecker(BaseASTVisitor):
    """
    Performs type inference and checking on MathViz AST.

    This visitor traverses the AST, inferring types for expressions,
    checking type consistency for declarations and assignments,
    and reporting any type errors found.

    Usage:
        checker = TypeChecker()
        errors = checker.check(program)
        if errors:
            for error in errors:
                print(error)
    """

    def __init__(self, source: str = "", filename: str = "<input>") -> None:
        """
        Initialize the type checker with empty state.

        Args:
            source: The source code being checked (for rich diagnostics)
            filename: The filename for error reporting
        """
        self.symbol_table = SymbolTable()
        self.function_signatures: dict[str, FunctionSignature] = {}
        self.class_types: dict[str, ClassType] = {}
        self.errors: list[MathVizTypeError] = []
        self.conversions: list[ConversionInfo] = []

        # OOP type registries
        self.struct_types: dict[str, StructType] = {}
        self.trait_types: dict[str, TraitType] = {}
        self.enum_types: dict[str, EnumType] = {}
        self.impl_blocks: dict[str, list[ImplBlock]] = {}  # type_name -> impl blocks

        # Rich diagnostics support
        self.source = source
        self.filename = filename
        self.diagnostics: list[Diagnostic] = []
        self._emitter: Optional[DiagnosticEmitter] = None

        # Type inference state
        self._unknown_type_counter = 0
        self._type_var_counter = 0
        self._current_function_return_type: Optional[Type] = None
        self._in_loop = False

        # Generic type parameter scope
        # Stack of type parameter scopes for nested generics
        self._type_param_scopes: list[dict[str, TypeVariable]] = []

        # Generic type registries
        self.generic_struct_types: dict[str, GenericStructType] = {}
        self.generic_enum_types: dict[str, GenericEnumType] = {}
        self.generic_function_signatures: dict[str, GenericFunctionType] = {}

        # Trait implementation registry: type_name -> set of implemented traits
        self.trait_implementations: dict[str, set[str]] = {}

        # Operator overloading registry: (type_name, trait_name) -> (method_signature, output_type)
        # For binary operators, trait_name might include type arg (e.g., "Mul<Float>")
        self.operator_impls: dict[tuple[str, str], tuple[FunctionType, Optional[Type]]] = {}

        # Initialize built-in functions and types
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in functions and types."""
        # Mathematical functions
        self._register_builtin_function("abs", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("sqrt", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("sin", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("cos", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("tan", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("exp", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("log", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("log10", [("x", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("pow", [("base", FLOAT_TYPE), ("exp", FLOAT_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("floor", [("x", FLOAT_TYPE)], INT_TYPE)
        self._register_builtin_function("ceil", [("x", FLOAT_TYPE)], INT_TYPE)
        self._register_builtin_function("round", [("x", FLOAT_TYPE)], INT_TYPE)

        # Collection functions
        self._register_builtin_function("len", [("x", ANY_TYPE)], INT_TYPE)
        self._register_builtin_function("sum", [("x", ANY_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("min", [("x", ANY_TYPE)], ANY_TYPE)
        self._register_builtin_function("max", [("x", ANY_TYPE)], ANY_TYPE)
        self._register_builtin_function("sorted", [("x", ANY_TYPE)], ANY_TYPE)
        self._register_builtin_function("reversed", [("x", ANY_TYPE)], ANY_TYPE)
        self._register_builtin_function("enumerate", [("x", ANY_TYPE)], ANY_TYPE)
        self._register_builtin_function("zip", [("a", ANY_TYPE), ("b", ANY_TYPE)], ANY_TYPE)
        self._register_builtin_function("range", [("stop", INT_TYPE)], RANGE_TYPE)

        # Type conversion functions
        self._register_builtin_function("int", [("x", ANY_TYPE)], INT_TYPE)
        self._register_builtin_function("float", [("x", ANY_TYPE)], FLOAT_TYPE)
        self._register_builtin_function("str", [("x", ANY_TYPE)], STRING_TYPE)
        self._register_builtin_function("bool", [("x", ANY_TYPE)], BOOL_TYPE)
        self._register_builtin_function("list", [("x", ANY_TYPE)], GenericTypeInstance("List", (ANY_TYPE,)))
        self._register_builtin_function("set", [("x", ANY_TYPE)], GenericTypeInstance("Set", (ANY_TYPE,)))
        self._register_builtin_function("dict", [("x", ANY_TYPE)], GenericTypeInstance("Dict", (ANY_TYPE, ANY_TYPE)))

        # Array/Vector/Matrix constructors
        self._register_builtin_function("Vec", [("data", ANY_TYPE)], VEC_TYPE)
        self._register_builtin_function("Mat", [("data", ANY_TYPE)], MAT_TYPE)
        self._register_builtin_function("Array", [("data", ANY_TYPE)], ARRAY_TYPE)
        self._register_builtin_function("zeros", [("shape", ANY_TYPE)], ARRAY_TYPE)
        self._register_builtin_function("ones", [("shape", ANY_TYPE)], ARRAY_TYPE)
        self._register_builtin_function("eye", [("n", INT_TYPE)], MAT_TYPE)
        self._register_builtin_function("linspace", [("start", FLOAT_TYPE), ("stop", FLOAT_TYPE), ("num", INT_TYPE)], VEC_TYPE)
        self._register_builtin_function("arange", [("stop", FLOAT_TYPE)], VEC_TYPE)

        # Manim objects (simplified)
        manim_object_type = ClassType("ManimObject")
        self._register_builtin_function("Circle", [], manim_object_type)
        self._register_builtin_function("Square", [], manim_object_type)
        self._register_builtin_function("Line", [("start", VEC_TYPE), ("end", VEC_TYPE)], manim_object_type)
        self._register_builtin_function("Text", [("text", STRING_TYPE)], manim_object_type)
        self._register_builtin_function("MathTex", [("tex", STRING_TYPE)], manim_object_type)
        self._register_builtin_function("Axes", [], manim_object_type)

        # Manim animations
        animation_type = ClassType("Animation")
        self._register_builtin_function("Create", [("mobject", manim_object_type)], animation_type)
        self._register_builtin_function("FadeIn", [("mobject", manim_object_type)], animation_type)
        self._register_builtin_function("FadeOut", [("mobject", manim_object_type)], animation_type)
        self._register_builtin_function("Transform", [("source", manim_object_type), ("target", manim_object_type)], animation_type)
        self._register_builtin_function("Write", [("mobject", manim_object_type)], animation_type)

        # Register class types
        self.class_types["ManimObject"] = manim_object_type
        self.class_types["Animation"] = animation_type

        # Register built-in constants
        self._register_builtin_constants()

    def _register_builtin_constants(self) -> None:
        """Register built-in mathematical constants."""
        # Mathematical constants (Float)
        float_constants = [
            "PI", "E", "TAU", "PHI", "SQRT2", "SQRT3",
            "LN2", "LN10", "LOG2E", "LOG10E",
            "INF", "NEG_INF", "NAN",
        ]
        for const in float_constants:
            self.symbol_table.define(const, FLOAT_TYPE, is_mutable=False)

        # Boolean constants
        self.symbol_table.define("TRUE", BOOL_TYPE, is_mutable=False)
        self.symbol_table.define("FALSE", BOOL_TYPE, is_mutable=False)

    def _register_builtin_function(
        self,
        name: str,
        params: list[tuple[str, Type]],
        return_type: Type,
    ) -> None:
        """Register a built-in function signature."""
        param_names = [p[0] for p in params]
        param_types = [p[1] for p in params]
        self.function_signatures[name] = FunctionSignature(
            name=name,
            param_types=param_types,
            param_names=param_names,
            return_type=return_type,
            has_defaults=[],
        )
        # Also add to symbol table as a function type
        self.symbol_table.define(name, FunctionType(
            param_types=tuple(param_types),
            return_type=return_type,
            param_names=tuple(param_names),
        ))

    def _new_unknown_type(self) -> UnknownType:
        """Create a new unique unknown type for inference."""
        self._unknown_type_counter += 1
        return UnknownType(self._unknown_type_counter)

    def _get_emitter(self) -> DiagnosticEmitter:
        """Get or create the diagnostic emitter."""
        if self._emitter is None:
            self._emitter = DiagnosticEmitter(self.source, self.filename)
        return self._emitter

    def _location_to_span(self, location: Optional[SourceLocation],
                          length: int = 1) -> Optional[SourceSpan]:
        """Convert a SourceLocation to a SourceSpan."""
        if location is None:
            return None
        return SourceSpan.from_location(
            location.line,
            location.column,
            length,
            location.filename or self.filename,
        )

    def _error(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
    ) -> None:
        """Record a type error (legacy simple error interface)."""
        self.errors.append(MathVizTypeError(message, location))

    # Alias for _error (some code paths use _emit_error)
    _emit_error = _error

    def _error_undefined_variable(
        self,
        name: str,
        location: Optional[SourceLocation],
    ) -> None:
        """Record an undefined variable error with suggestions."""
        span = self._location_to_span(location, len(name))
        if span and self.source:
            # Get all visible variable names for suggestions
            candidates = self.symbol_table.get_all_symbol_names()
            definitions = self.symbol_table.get_all_variable_info()

            diagnostic = create_undefined_variable_diagnostic(
                self._get_emitter(),
                name,
                span,
                candidates,
                definitions,
            )
            self.diagnostics.append(diagnostic)

        # Also add to legacy errors for compatibility
        self.errors.append(MathVizTypeError(f"Undefined variable: '{name}'", location))

    def _error_undefined_function(
        self,
        name: str,
        location: Optional[SourceLocation],
    ) -> None:
        """Record an undefined function error with suggestions."""
        span = self._location_to_span(location, len(name))
        if span and self.source:
            # Get all function names for suggestions
            candidates = list(self.function_signatures.keys())

            diagnostic = create_undefined_function_diagnostic(
                self._get_emitter(),
                name,
                span,
                candidates,
            )
            self.diagnostics.append(diagnostic)

        self.errors.append(MathVizTypeError(f"Undefined function: '{name}'", location))

    def _error_type_mismatch(
        self,
        expected: Type,
        actual: Type,
        location: Optional[SourceLocation],
        context: str = "",
    ) -> None:
        """Record a type mismatch error with conversion suggestions."""
        span = self._location_to_span(location)
        if span and self.source:
            diagnostic = create_type_mismatch_diagnostic(
                self._get_emitter(),
                str(expected),
                str(actual),
                span,
                context,
            )
            self.diagnostics.append(diagnostic)

        msg = f"Expected type '{expected}', got '{actual}'"
        if context:
            msg = f"{msg} {context}"
        self.errors.append(MathVizTypeError(msg, location))

    def _error_wrong_args(
        self,
        func_name: str,
        expected_min: int,
        expected_max: int,
        actual: int,
        location: Optional[SourceLocation],
    ) -> None:
        """Record a wrong number of arguments error."""
        span = self._location_to_span(location)
        if span and self.source:
            diagnostic = create_wrong_arguments_diagnostic(
                self._get_emitter(),
                func_name,
                expected_min,
                expected_max,
                actual,
                span,
            )
            self.diagnostics.append(diagnostic)

        if expected_min == expected_max:
            expected_str = str(expected_min)
        else:
            expected_str = f"{expected_min}-{expected_max}"
        self.errors.append(MathVizTypeError(
            f"Function '{func_name}' expects {expected_str} argument(s), got {actual}",
            location,
        ))

    def _record_conversion(
        self,
        from_type: Type,
        to_type: Type,
        conversion: TypeConversion,
        location: Optional[SourceLocation] = None,
    ) -> None:
        """Record an implicit type conversion."""
        if conversion != TypeConversion.NONE:
            self.conversions.append(ConversionInfo(from_type, to_type, conversion, location))

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def check(self, program: Program, source: str = "",
              filename: str = "<input>") -> list[MathVizTypeError]:
        """
        Type check the entire program, return list of errors.

        Args:
            program: The Program AST node to check
            source: Optional source code for rich diagnostics
            filename: Optional filename for error reporting

        Returns:
            A list of TypeError instances for any type errors found
        """
        self.errors = []
        self.conversions = []
        self.diagnostics = []

        # Set up rich diagnostics if source provided
        if source:
            self.source = source
            self.filename = filename
            self._emitter = DiagnosticEmitter(source, filename)

        self.visit(program)
        return self.errors

    def get_diagnostics(self) -> list[Diagnostic]:
        """Get all rich diagnostics emitted during type checking."""
        return self.diagnostics

    def render_diagnostics(self, use_color: bool = True) -> str:
        """Render all diagnostics as formatted strings."""
        if self._emitter:
            return self._emitter.render_all(use_color)
        return ""

    def infer_type(self, expr: Expression) -> Type:
        """
        Infer the type of an expression.

        Args:
            expr: The expression to infer the type of

        Returns:
            The inferred Type
        """
        return self._infer_expression_type(expr)

    def get_conversions(self) -> list[ConversionInfo]:
        """Get all implicit conversions detected during type checking."""
        return self.conversions

    # -------------------------------------------------------------------------
    # Type Annotation Resolution
    # -------------------------------------------------------------------------

    def _resolve_type_annotation(self, annotation: TypeAnnotation) -> Type:
        """Convert an AST type annotation to a Type object."""
        if isinstance(annotation, SimpleType):
            return self._resolve_simple_type(annotation.name)
        elif isinstance(annotation, GenericType):
            return self._resolve_generic_type(annotation)
        elif isinstance(annotation, ASTFunctionType):
            return self._resolve_function_type(annotation)
        else:
            self._error(f"Unknown type annotation: {annotation}", annotation.location)
            return UNKNOWN_TYPE

    def _resolve_simple_type(self, name: str) -> Type:
        """Resolve a simple type name to a Type object."""
        # Check for type variable in scope first (for generics)
        type_var = self._lookup_type_param(name)
        if type_var is not None:
            return type_var

        # Primitive types
        primitives = {
            "Int": INT_TYPE,
            "Float": FLOAT_TYPE,
            "Bool": BOOL_TYPE,
            "String": STRING_TYPE,
            "None": NONE_TYPE,
            "Void": NONE_TYPE,
        }
        if name in primitives:
            return primitives[name]

        # Mathematical types
        if name == "Vec":
            return VEC_TYPE
        if name == "Mat":
            return MAT_TYPE
        if name == "Array":
            return ARRAY_TYPE

        # Any type
        if name == "Any":
            return ANY_TYPE

        # Check for user-defined class types
        if name in self.class_types:
            return self.class_types[name]

        # Check for struct types
        if name in self.struct_types:
            return self.struct_types[name]

        # Check for enum types
        if name in self.enum_types:
            return self.enum_types[name]

        # Unknown type - might be defined later or external
        return ClassType(name)

    def _enter_type_param_scope(self, type_params: tuple) -> dict[str, TypeVariable]:
        """
        Enter a new type parameter scope for generic type checking.

        Args:
            type_params: TypeParameter nodes from the AST

        Returns:
            A dict mapping type parameter names to TypeVariable instances
        """
        from mathviz.compiler.ast_nodes import TypeParameter as ASTTypeParameter

        scope: dict[str, TypeVariable] = {}
        for param in type_params:
            self._type_var_counter += 1
            bounds = param.bounds if isinstance(param, ASTTypeParameter) else ()
            type_var = TypeVariable(param.name, bounds, self._type_var_counter)
            scope[param.name] = type_var
        self._type_param_scopes.append(scope)
        return scope

    def _exit_type_param_scope(self) -> None:
        """Exit the current type parameter scope."""
        if self._type_param_scopes:
            self._type_param_scopes.pop()

    def _lookup_type_param(self, name: str) -> Optional[TypeVariable]:
        """
        Look up a type parameter in the current scope stack.

        Searches from innermost to outermost scope.
        """
        for scope in reversed(self._type_param_scopes):
            if name in scope:
                return scope[name]
        return None

    def _current_type_params(self) -> dict[str, TypeVariable]:
        """Get all type parameters currently in scope."""
        result: dict[str, TypeVariable] = {}
        for scope in self._type_param_scopes:
            result.update(scope)
        return result

    def _resolve_generic_type(self, annotation: GenericType) -> Type:
        """Resolve a generic type annotation to a Type object."""
        type_args = tuple(self._resolve_type_annotation(arg) for arg in annotation.type_args)

        # Handle collection types
        if annotation.base in ("List", "Set"):
            if len(type_args) != 1:
                self._error(
                    f"{annotation.base} requires exactly 1 type argument, got {len(type_args)}",
                    annotation.location,
                )
                return GenericTypeInstance(annotation.base, (ANY_TYPE,))
            return GenericTypeInstance(annotation.base, type_args)

        if annotation.base == "Dict":
            if len(type_args) != 2:
                self._error(
                    f"Dict requires exactly 2 type arguments, got {len(type_args)}",
                    annotation.location,
                )
                return GenericTypeInstance("Dict", (ANY_TYPE, ANY_TYPE))
            return GenericTypeInstance("Dict", type_args)

        # Handle array types with element type
        if annotation.base in ("Array", "Vec", "Mat"):
            if len(type_args) != 1:
                self._error(
                    f"{annotation.base} requires exactly 1 type argument, got {len(type_args)}",
                    annotation.location,
                )
                return ArrayType(annotation.base)
            return ArrayType(annotation.base, type_args[0])

        # Handle Tuple type
        if annotation.base == "Tuple":
            return TupleType(type_args)

        # Handle Optional type
        if annotation.base == "Optional":
            if len(type_args) != 1:
                self._error(
                    f"Optional requires exactly 1 type argument, got {len(type_args)}",
                    annotation.location,
                )
                return OptionalType(ANY_TYPE)
            return OptionalType(type_args[0])

        # Handle Result type
        if annotation.base == "Result":
            if len(type_args) != 2:
                self._error(
                    f"Result requires exactly 2 type arguments (ok_type, err_type), got {len(type_args)}",
                    annotation.location,
                )
                return ResultType(ANY_TYPE, STRING_TYPE)
            return ResultType(type_args[0], type_args[1])

        # Generic user type
        return GenericTypeInstance(annotation.base, type_args)

    def _resolve_function_type(self, annotation: ASTFunctionType) -> Type:
        """Resolve a function type annotation to a Type object."""
        param_types = tuple(
            self._resolve_type_annotation(param) for param in annotation.param_types
        )
        return_type = self._resolve_type_annotation(annotation.return_type)
        return FunctionType(param_types, return_type)

    # -------------------------------------------------------------------------
    # Type Inference for Expressions
    # -------------------------------------------------------------------------

    def _infer_expression_type(self, expr: Expression) -> Type:
        """Infer the type of an expression by dispatching to specific handlers."""
        if isinstance(expr, IntegerLiteral):
            return INT_TYPE
        elif isinstance(expr, FloatLiteral):
            return FLOAT_TYPE
        elif isinstance(expr, StringLiteral):
            return STRING_TYPE
        elif isinstance(expr, FString):
            return self._infer_fstring_type(expr)
        elif isinstance(expr, BooleanLiteral):
            return BOOL_TYPE
        elif isinstance(expr, NoneLiteral):
            return NONE_TYPE
        elif isinstance(expr, Identifier):
            return self._infer_identifier_type(expr)
        elif isinstance(expr, ListLiteral):
            return self._infer_list_literal_type(expr)
        elif isinstance(expr, SetLiteral):
            return self._infer_set_literal_type(expr)
        elif isinstance(expr, DictLiteral):
            return self._infer_dict_literal_type(expr)
        elif isinstance(expr, BinaryExpression):
            return self._infer_binary_expression_type(expr)
        elif isinstance(expr, UnaryExpression):
            return self._infer_unary_expression_type(expr)
        elif isinstance(expr, CallExpression):
            return self._infer_call_expression_type(expr)
        elif isinstance(expr, MemberAccess):
            return self._infer_member_access_type(expr)
        elif isinstance(expr, IndexExpression):
            return self._infer_index_expression_type(expr)
        elif isinstance(expr, ConditionalExpression):
            return self._infer_conditional_expression_type(expr)
        elif isinstance(expr, LambdaExpression):
            return self._infer_lambda_expression_type(expr)
        elif isinstance(expr, RangeExpression):
            return self._infer_range_expression_type(expr)
        elif isinstance(expr, MatchExpression):
            return self._infer_match_expression_type(expr)
        elif isinstance(expr, TupleLiteral):
            return self._infer_tuple_literal_type(expr)
        elif isinstance(expr, SomeExpression):
            return self._infer_some_expression_type(expr)
        elif isinstance(expr, OkExpression):
            return self._infer_ok_expression_type(expr)
        elif isinstance(expr, ErrExpression):
            return self._infer_err_expression_type(expr)
        elif isinstance(expr, UnwrapExpression):
            return self._infer_unwrap_expression_type(expr)
        elif isinstance(expr, SelfExpression):
            return self.visit_self_expression(expr)
        elif isinstance(expr, EnumVariantAccess):
            return self.visit_enum_variant_access(expr)
        elif isinstance(expr, StructLiteral):
            return self.visit_struct_literal(expr)
        else:
            self._error(f"Cannot infer type of expression: {type(expr).__name__}", expr.location)
            return UNKNOWN_TYPE

    def _infer_identifier_type(self, expr: Identifier) -> Type:
        """Infer the type of an identifier by looking it up in the symbol table."""
        type_ = self.symbol_table.lookup(expr.name)
        if type_ is None:
            # Check if it's a type name (enum, struct, class)
            if expr.name in self.enum_types:
                return self.enum_types[expr.name]
            if expr.name in self.struct_types:
                return self.struct_types[expr.name]
            if expr.name in self.class_types:
                return self.class_types[expr.name]
            self._error_undefined_variable(expr.name, expr.location)
            return UNKNOWN_TYPE
        return type_

    def _infer_fstring_type(self, expr: FString) -> Type:
        """
        Infer the type of an f-string (always String).

        Also type-check all expressions within the f-string.
        """
        # Type-check all expression parts
        for part in expr.parts:
            if isinstance(part, FStringExpression):
                # Infer the type of the expression (validates it)
                self._infer_expression_type(part.expression)
        # F-strings always evaluate to String
        return STRING_TYPE

    def _infer_list_literal_type(self, expr: ListLiteral) -> Type:
        """Infer the type of a list literal."""
        if not expr.elements:
            return GenericTypeInstance("List", (UNKNOWN_TYPE,))

        element_types = [self._infer_expression_type(elem) for elem in expr.elements]
        unified_type = self._unify_types(element_types, expr.location)
        return GenericTypeInstance("List", (unified_type,))

    def _infer_set_literal_type(self, expr: SetLiteral) -> Type:
        """Infer the type of a set literal."""
        if not expr.elements:
            return GenericTypeInstance("Set", (UNKNOWN_TYPE,))

        element_types = [self._infer_expression_type(elem) for elem in expr.elements]
        unified_type = self._unify_types(element_types, expr.location)
        return GenericTypeInstance("Set", (unified_type,))

    def _infer_dict_literal_type(self, expr: DictLiteral) -> Type:
        """Infer the type of a dictionary literal."""
        if not expr.pairs:
            return GenericTypeInstance("Dict", (UNKNOWN_TYPE, UNKNOWN_TYPE))

        key_types = [self._infer_expression_type(k) for k, _ in expr.pairs]
        value_types = [self._infer_expression_type(v) for _, v in expr.pairs]

        key_type = self._unify_types(key_types, expr.location)
        value_type = self._unify_types(value_types, expr.location)

        return GenericTypeInstance("Dict", (key_type, value_type))

    def _infer_binary_expression_type(self, expr: BinaryExpression) -> Type:
        """Infer the type of a binary expression based on operator and operands."""
        left_type = self._infer_expression_type(expr.left)
        right_type = self._infer_expression_type(expr.right)
        op = expr.operator

        # First, check for operator overloading if the left operand is a user-defined type
        overload_result = self._check_binary_operator_overload(left_type, right_type, op)
        if overload_result is not None:
            return overload_result

        # Arithmetic operators return numeric types
        if op in (
            BinaryOperator.ADD, BinaryOperator.SUB, BinaryOperator.MUL,
            BinaryOperator.DIV, BinaryOperator.MOD, BinaryOperator.POW,
            BinaryOperator.FLOOR_DIV,
        ):
            return self._infer_arithmetic_result_type(left_type, right_type, op, expr.location)

        # Comparison operators return Bool
        if op in (
            BinaryOperator.EQ, BinaryOperator.NE, BinaryOperator.LT,
            BinaryOperator.GT, BinaryOperator.LE, BinaryOperator.GE,
        ):
            self._check_comparison_operands(left_type, right_type, op, expr.location)
            return BOOL_TYPE

        # Logical operators return Bool
        if op in (BinaryOperator.AND, BinaryOperator.OR):
            self._check_boolean_operand(left_type, "left operand of logical operation", expr.left.location)
            self._check_boolean_operand(right_type, "right operand of logical operation", expr.right.location)
            return BOOL_TYPE

        # Membership operators return Bool
        if op in (
            BinaryOperator.IN, BinaryOperator.NOT_IN,
            BinaryOperator.ELEMENT_OF, BinaryOperator.NOT_ELEMENT_OF,
        ):
            return BOOL_TYPE

        # Set relationship operators return Bool
        if op in (
            BinaryOperator.SUBSET, BinaryOperator.SUPERSET,
            BinaryOperator.PROPER_SUBSET, BinaryOperator.PROPER_SUPERSET,
        ):
            self._check_set_operand(left_type, "left operand of set operation", expr.left.location)
            self._check_set_operand(right_type, "right operand of set operation", expr.right.location)
            return BOOL_TYPE

        # Set operations return Set
        if op in (BinaryOperator.UNION, BinaryOperator.INTERSECTION, BinaryOperator.SET_DIFF):
            self._check_set_operand(left_type, "left operand of set operation", expr.left.location)
            self._check_set_operand(right_type, "right operand of set operation", expr.right.location)
            return self._unify_types([left_type, right_type], expr.location)

        self._error(f"Unknown binary operator: {op}", expr.location)
        return UNKNOWN_TYPE

    def _check_binary_operator_overload(
        self,
        left_type: Type,
        right_type: Type,
        op: BinaryOperator,
    ) -> Optional[Type]:
        """
        Check if a binary operator has an overloaded implementation for the given types.

        Returns the result type if an overload exists, None otherwise.
        """
        # Get the type name for lookup
        type_name = self._get_type_name_for_operator(left_type)
        if type_name is None:
            return None

        # Get the trait name for this operator
        trait_name = get_trait_for_binary_operator(op)
        if trait_name is None:
            return None

        # Check for exact trait match (e.g., impl Add for Vector)
        key = (type_name, trait_name)
        if key in self.operator_impls:
            _, output_type = self.operator_impls[key]
            return output_type

        # Check for generic trait match (e.g., impl Mul<Float> for Vector)
        right_type_name = self._get_type_name_for_operator(right_type) or str(right_type)
        generic_key = (type_name, f"{trait_name}<{right_type_name}>")
        if generic_key in self.operator_impls:
            _, output_type = self.operator_impls[generic_key]
            return output_type

        return None

    def _get_type_name_for_operator(self, type_: Type) -> Optional[str]:
        """Get the type name for operator overload lookup."""
        if isinstance(type_, StructType):
            return type_.name
        if isinstance(type_, ClassType):
            return type_.name
        if isinstance(type_, EnumType):
            return type_.name
        if isinstance(type_, PrimitiveType):
            return type_.name
        return None

    def _infer_arithmetic_result_type(
        self,
        left_type: Type,
        right_type: Type,
        op: BinaryOperator,
        location: Optional[SourceLocation],
    ) -> Type:
        """Determine the result type of an arithmetic operation."""
        # Check that both operands are numeric
        if not self._is_numeric_type(left_type):
            self._error(
                f"Left operand of '{op.name.lower()}' must be numeric, got {left_type}",
                location,
            )
            return UNKNOWN_TYPE

        if not self._is_numeric_type(right_type):
            self._error(
                f"Right operand of '{op.name.lower()}' must be numeric, got {right_type}",
                location,
            )
            return UNKNOWN_TYPE

        # Division always returns Float
        if op == BinaryOperator.DIV:
            if left_type == INT_TYPE:
                self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, location)
            return FLOAT_TYPE

        # Floor division returns Int if both operands are Int
        if op == BinaryOperator.FLOOR_DIV:
            if left_type == INT_TYPE and right_type == INT_TYPE:
                return INT_TYPE
            return FLOAT_TYPE

        # For other arithmetic ops, promote to Float if either operand is Float
        if left_type == FLOAT_TYPE or right_type == FLOAT_TYPE:
            if left_type == INT_TYPE:
                self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, location)
            if right_type == INT_TYPE:
                self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, location)
            return FLOAT_TYPE

        # Array operations
        if isinstance(left_type, ArrayType) or isinstance(right_type, ArrayType):
            # Return the array type (element-wise operations)
            if isinstance(left_type, ArrayType):
                return left_type
            return right_type

        # Both are Int
        return INT_TYPE

    def _is_numeric_type(self, type_: Type) -> bool:
        """Check if a type is numeric."""
        if isinstance(type_, PrimitiveType):
            return type_.is_numeric()
        if isinstance(type_, ArrayType):
            return type_.is_numeric()
        if isinstance(type_, (UnknownType, AnyType)):
            return True  # Assume numeric during inference
        return False

    def _check_comparison_operands(
        self,
        left_type: Type,
        right_type: Type,
        op: BinaryOperator,
        location: Optional[SourceLocation],
    ) -> None:
        """Check that comparison operands are compatible."""
        if not left_type.is_compatible_with(right_type) and not right_type.is_compatible_with(left_type):
            self._error(
                f"Cannot compare {left_type} with {right_type}",
                location,
            )

    def _check_boolean_operand(
        self,
        type_: Type,
        context: str,
        location: Optional[SourceLocation],
    ) -> None:
        """Check that an operand can be used as a boolean."""
        if type_ != BOOL_TYPE and not isinstance(type_, (UnknownType, AnyType)):
            self._error(f"Expected Bool for {context}, got {type_}", location)

    def _check_set_operand(
        self,
        type_: Type,
        context: str,
        location: Optional[SourceLocation],
    ) -> None:
        """Check that an operand is a set type."""
        if isinstance(type_, GenericTypeInstance) and type_.base_name == "Set":
            return
        if isinstance(type_, (UnknownType, AnyType)):
            return
        self._error(f"Expected Set for {context}, got {type_}", location)

    def _infer_unary_expression_type(self, expr: UnaryExpression) -> Type:
        """Infer the type of a unary expression."""
        operand_type = self._infer_expression_type(expr.operand)

        # First, check for operator overloading
        overload_result = self._check_unary_operator_overload(operand_type, expr.operator)
        if overload_result is not None:
            return overload_result

        if expr.operator == UnaryOperator.NEG:
            if not self._is_numeric_type(operand_type):
                self._error(
                    f"Unary '-' requires numeric operand, got {operand_type}",
                    expr.location,
                )
                return UNKNOWN_TYPE
            return operand_type

        if expr.operator == UnaryOperator.POS:
            if not self._is_numeric_type(operand_type):
                self._error(
                    f"Unary '+' requires numeric operand, got {operand_type}",
                    expr.location,
                )
                return UNKNOWN_TYPE
            return operand_type

        if expr.operator == UnaryOperator.NOT:
            self._check_boolean_operand(operand_type, "operand of 'not'", expr.operand.location)
            return BOOL_TYPE

        return UNKNOWN_TYPE

    def _check_unary_operator_overload(
        self,
        operand_type: Type,
        op: UnaryOperator,
    ) -> Optional[Type]:
        """
        Check if a unary operator has an overloaded implementation for the given type.

        Returns the result type if an overload exists, None otherwise.
        """
        # Get the type name for lookup
        type_name = self._get_type_name_for_operator(operand_type)
        if type_name is None:
            return None

        # Get the trait name for this operator
        trait_name = get_trait_for_unary_operator(op)
        if trait_name is None:
            return None

        # Check for trait match (e.g., impl Neg for Vector)
        key = (type_name, trait_name)
        if key in self.operator_impls:
            _, output_type = self.operator_impls[key]
            return output_type

        return None

    def _infer_call_expression_type(self, expr: CallExpression) -> Type:
        """Infer the type of a function call expression."""
        # Get the callee type
        if isinstance(expr.callee, Identifier):
            callee_name = expr.callee.name

            # Check function signatures first
            if callee_name in self.function_signatures:
                sig = self.function_signatures[callee_name]
                self._check_function_call_args(sig, expr.arguments, expr.location)
                return sig.return_type

            # Check symbol table for function type
            callee_type = self.symbol_table.lookup(callee_name)
            if callee_type is None:
                self._error_undefined_function(callee_name, expr.callee.location)
                return UNKNOWN_TYPE

            if isinstance(callee_type, FunctionType):
                self._check_call_arguments(callee_type, expr.arguments, expr.location)
                return callee_type.return_type

            # Could be a class constructor
            if callee_name in self.class_types:
                return self.class_types[callee_name]

            # Unknown callable - assume Any
            return ANY_TYPE

        # Method call or complex callee
        callee_type = self._infer_expression_type(expr.callee)

        if isinstance(callee_type, FunctionType):
            self._check_call_arguments(callee_type, expr.arguments, expr.location)
            return callee_type.return_type

        if isinstance(callee_type, (UnknownType, AnyType)):
            return ANY_TYPE

        self._error(f"Cannot call non-function type: {callee_type}", expr.location)
        return UNKNOWN_TYPE

    def _check_function_call_args(
        self,
        sig: FunctionSignature,
        args: Sequence[Expression],
        location: Optional[SourceLocation],
    ) -> None:
        """Check that function call arguments match the signature."""
        num_args = len(args)

        if num_args < sig.min_args:
            self._error_wrong_args(sig.name, sig.min_args, sig.max_args, num_args, location)
            return

        if num_args > sig.max_args:
            self._error_wrong_args(sig.name, sig.min_args, sig.max_args, num_args, location)
            return

        # Check each argument type
        for i, (arg, expected_type) in enumerate(zip(args, sig.param_types)):
            arg_type = self._infer_expression_type(arg)
            if not arg_type.is_compatible_with(expected_type):
                param_name = sig.param_names[i] if i < len(sig.param_names) else f"argument {i+1}"
                context = f"in argument '{param_name}' of function '{sig.name}'"
                self._error_type_mismatch(expected_type, arg_type, arg.location, context)
            elif arg_type != expected_type and not isinstance(expected_type, (UnknownType, AnyType)):
                # Record implicit conversion
                if arg_type == INT_TYPE and expected_type == FLOAT_TYPE:
                    self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, arg.location)

    def _check_call_arguments(
        self,
        func_type: FunctionType,
        args: Sequence[Expression],
        location: Optional[SourceLocation],
    ) -> None:
        """Check that call arguments match a function type."""
        expected = len(func_type.param_types)
        actual = len(args)

        if expected != actual:
            self._error(
                f"Expected {expected} arguments, got {actual}",
                location,
            )
            return

        for i, (arg, expected_type) in enumerate(zip(args, func_type.param_types)):
            arg_type = self._infer_expression_type(arg)
            if not arg_type.is_compatible_with(expected_type):
                self._error(
                    f"Argument {i+1} expects {expected_type}, got {arg_type}",
                    arg.location,
                )

    def _infer_member_access_type(self, expr: MemberAccess) -> Type:
        """Infer the type of a member access expression."""
        object_type = self._infer_expression_type(expr.object)

        # Handle array/list length
        if expr.member == "length" or expr.member == "size":
            if isinstance(object_type, (GenericTypeInstance, ArrayType)):
                return INT_TYPE

        # Handle array/list methods
        if isinstance(object_type, GenericTypeInstance):
            if object_type.base_name == "List":
                element_type = object_type.type_args[0] if object_type.type_args else ANY_TYPE
                if expr.member == "append":
                    return FunctionType((element_type,), NONE_TYPE)
                if expr.member == "pop":
                    return FunctionType((), element_type)
                if expr.member == "sort":
                    return FunctionType((), NONE_TYPE)
                if expr.member == "reverse":
                    return FunctionType((), NONE_TYPE)
            elif object_type.base_name == "Dict":
                key_type = object_type.type_args[0] if len(object_type.type_args) > 0 else ANY_TYPE
                value_type = object_type.type_args[1] if len(object_type.type_args) > 1 else ANY_TYPE
                if expr.member == "keys":
                    return FunctionType((), GenericTypeInstance("List", (key_type,)))
                if expr.member == "values":
                    return FunctionType((), GenericTypeInstance("List", (value_type,)))
                if expr.member == "items":
                    return FunctionType((), GenericTypeInstance("List", (ANY_TYPE,)))
                if expr.member == "get":
                    return FunctionType((key_type,), value_type)

        # Handle array methods
        if isinstance(object_type, ArrayType):
            if expr.member in ("shape", "size"):
                return GenericTypeInstance("Tuple", (INT_TYPE,))
            if expr.member == "T":
                return object_type  # Transpose has same type
            if expr.member == "dot":
                return FunctionType((object_type,), object_type)
            if expr.member == "sum":
                return FunctionType((), object_type.element_type or FLOAT_TYPE)
            if expr.member == "mean":
                return FunctionType((), FLOAT_TYPE)

        # For class types, we'd need to look up the class definition
        # For now, return Any for unknown members
        if isinstance(object_type, ClassType):
            return ANY_TYPE

        # Handle struct field access
        if isinstance(object_type, StructType):
            field_type = object_type.get_field_type(expr.member)
            if field_type:
                return field_type
            # Check for methods from impl blocks
            type_name = object_type.name
            if type_name in self.impl_blocks:
                for impl_block in self.impl_blocks[type_name]:
                    for method in impl_block.methods:
                        if method.name == expr.member:
                            # Return method signature as function type
                            param_types = tuple(
                                self._resolve_type_annotation(p.type_annotation)
                                if p.type_annotation else ANY_TYPE
                                for p in method.parameters
                            )
                            return_type = (
                                self._resolve_type_annotation(method.return_type)
                                if method.return_type else NONE_TYPE
                            )
                            return FunctionType(param_types, return_type)
            self._error(
                f"Unknown field or method '{expr.member}' on struct '{object_type.name}'",
                expr.location,
            )
            return UNKNOWN_TYPE

        if isinstance(object_type, (UnknownType, AnyType)):
            return ANY_TYPE

        # Could not determine member type
        return ANY_TYPE

    def _infer_index_expression_type(self, expr: IndexExpression) -> Type:
        """Infer the type of an index expression (subscript)."""
        object_type = self._infer_expression_type(expr.object)
        index_type = self._infer_expression_type(expr.index)

        # List indexing
        if isinstance(object_type, GenericTypeInstance):
            if object_type.base_name == "List":
                if not index_type.is_compatible_with(INT_TYPE):
                    self._error(
                        f"List index must be Int, got {index_type}",
                        expr.index.location,
                    )
                return object_type.type_args[0] if object_type.type_args else ANY_TYPE

            if object_type.base_name == "Dict":
                key_type = object_type.type_args[0] if object_type.type_args else ANY_TYPE
                value_type = object_type.type_args[1] if len(object_type.type_args) > 1 else ANY_TYPE
                if not index_type.is_compatible_with(key_type):
                    self._error(
                        f"Dict key must be {key_type}, got {index_type}",
                        expr.index.location,
                    )
                return value_type

        # Array indexing
        if isinstance(object_type, ArrayType):
            if not index_type.is_compatible_with(INT_TYPE):
                # Could also be slice or tuple for multi-dimensional
                pass
            return object_type.element_type or FLOAT_TYPE

        # String indexing
        if object_type == STRING_TYPE:
            if not index_type.is_compatible_with(INT_TYPE):
                self._error(
                    f"String index must be Int, got {index_type}",
                    expr.index.location,
                )
            return STRING_TYPE

        if isinstance(object_type, (UnknownType, AnyType)):
            return ANY_TYPE

        self._error(f"Cannot index into type: {object_type}", expr.location)
        return UNKNOWN_TYPE

    def _infer_conditional_expression_type(self, expr: ConditionalExpression) -> Type:
        """Infer the type of a conditional (ternary) expression."""
        cond_type = self._infer_expression_type(expr.condition)
        self._check_boolean_operand(cond_type, "condition of conditional expression", expr.condition.location)

        then_type = self._infer_expression_type(expr.then_expr)
        else_type = self._infer_expression_type(expr.else_expr)

        # Result type is the unified type of both branches
        return self._unify_types([then_type, else_type], expr.location)

    def _infer_lambda_expression_type(self, expr: LambdaExpression) -> Type:
        """Infer the type of a lambda expression."""
        # Enter a new scope for lambda parameters
        self.symbol_table.enter_scope("lambda")

        param_types: list[Type] = []
        param_names: list[str] = []

        for param in expr.parameters:
            param_names.append(param.name)
            if param.type_annotation:
                param_type = self._resolve_type_annotation(param.type_annotation)
            else:
                param_type = self._new_unknown_type()
            param_types.append(param_type)
            self.symbol_table.define(param.name, param_type)

        # Infer body type
        if isinstance(expr.body, Block):
            self.visit(expr.body)
            return_type = NONE_TYPE  # Block lambda returns None unless explicit return
        else:
            return_type = self._infer_expression_type(expr.body)

        self.symbol_table.exit_scope()

        return FunctionType(
            param_types=tuple(param_types),
            return_type=return_type,
            param_names=tuple(param_names),
        )

    def _infer_range_expression_type(self, expr: RangeExpression) -> Type:
        """Infer the type of a range expression."""
        start_type = self._infer_expression_type(expr.start)
        end_type = self._infer_expression_type(expr.end)

        if not start_type.is_compatible_with(INT_TYPE):
            self._error(f"Range start must be Int, got {start_type}", expr.start.location)

        if not end_type.is_compatible_with(INT_TYPE):
            self._error(f"Range end must be Int, got {end_type}", expr.end.location)

        if expr.step:
            step_type = self._infer_expression_type(expr.step)
            if not step_type.is_compatible_with(INT_TYPE):
                self._error(f"Range step must be Int, got {step_type}", expr.step.location)

        return RANGE_TYPE

    def _infer_tuple_literal_type(self, expr: TupleLiteral) -> Type:
        """Infer the type of a tuple literal."""
        if not expr.elements:
            return EMPTY_TUPLE_TYPE
        element_types = tuple(self._infer_expression_type(e) for e in expr.elements)
        return TupleType(element_types)

    def _infer_some_expression_type(self, expr: SomeExpression) -> Type:
        """Infer the type of a Some(value) expression."""
        inner_type = self._infer_expression_type(expr.value)
        return OptionalType(inner_type)

    def _infer_ok_expression_type(self, expr: OkExpression) -> Type:
        """
        Infer the type of an Ok(value) expression.

        Note: The error type is inferred as UNKNOWN_TYPE and should be unified
        with context later during type inference/checking.
        """
        ok_type = self._infer_expression_type(expr.value)
        # Error type unknown without context
        return ResultType(ok_type, UNKNOWN_TYPE)

    def _infer_err_expression_type(self, expr: ErrExpression) -> Type:
        """
        Infer the type of an Err(error) expression.

        Note: The ok type is inferred as UNKNOWN_TYPE and should be unified
        with context later during type inference/checking.
        """
        err_type = self._infer_expression_type(expr.value)
        # Ok type unknown without context
        return ResultType(UNKNOWN_TYPE, err_type)

    def _infer_unwrap_expression_type(self, expr: UnwrapExpression) -> Type:
        """
        Infer the type of an unwrap expression (e.g., value?).

        For Optional[T], returns T.
        For Result[T, E], returns T (and propagates errors).
        """
        operand_type = self._infer_expression_type(expr.operand)

        if isinstance(operand_type, OptionalType):
            return operand_type.unwrap_type()

        if isinstance(operand_type, ResultType):
            return operand_type.unwrap_type()

        if isinstance(operand_type, (UnknownType, AnyType)):
            return UNKNOWN_TYPE

        self._error(
            f"Cannot unwrap type {operand_type}. Expected Optional[T] or Result[T, E]",
            expr.location,
        )
        return UNKNOWN_TYPE

    def _infer_match_expression_type(self, expr: MatchExpression) -> Type:
        """
        Infer the type of a match expression.

        The type is the unified type of all arm bodies.
        Also performs exhaustiveness checking.
        """
        subject_type = self._infer_expression_type(expr.subject)

        # Type check each arm and collect body types
        body_types: list[Type] = []
        has_wildcard = False
        wildcard_index: Optional[int] = None

        for i, arm in enumerate(expr.arms):
            # Enter a new scope for pattern bindings
            self.symbol_table.enter_scope(f"match:arm:{i}")

            # Bind pattern variables
            self._bind_pattern_variables(arm.pattern, subject_type)

            # Check for wildcard pattern
            if self._is_wildcard_pattern(arm.pattern):
                if has_wildcard:
                    self._error(
                        "Unreachable pattern: wildcard pattern already covered all cases",
                        arm.location,
                    )
                has_wildcard = True
                wildcard_index = i

            # Check if this pattern is reachable (after wildcard)
            if has_wildcard and wildcard_index is not None and i > wildcard_index:
                self._error(
                    f"Unreachable pattern: previous pattern at arm {wildcard_index + 1} "
                    "already matches all cases",
                    arm.location,
                )

            # Type check guard if present
            if arm.guard:
                guard_type = self._infer_expression_type(arm.guard)
                self._check_boolean_operand(guard_type, "match guard", arm.guard.location)

            # Type check body
            if isinstance(arm.body, Block):
                self.visit(arm.body)
                body_type = NONE_TYPE  # Block without return yields None
            else:
                body_type = self._infer_expression_type(arm.body)

            body_types.append(body_type)
            self.symbol_table.exit_scope()

        # Check exhaustiveness
        self._check_exhaustiveness(expr, subject_type, has_wildcard)

        # The match expression type is the unified type of all arms
        return self._unify_types(body_types, expr.location)

    def _bind_pattern_variables(self, pattern: Pattern, subject_type: Type) -> None:
        """Bind variables from a pattern to the appropriate types."""
        if isinstance(pattern, LiteralPattern):
            # Literal patterns don't bind variables
            pass
        elif isinstance(pattern, IdentifierPattern):
            if not pattern.is_wildcard:
                self.symbol_table.define(pattern.name, subject_type, pattern.location)
        elif isinstance(pattern, TuplePattern):
            # For tuples, we need to infer element types
            # Handle rest patterns specially
            rest_index = self._find_rest_pattern_index(pattern.elements)
            if rest_index is not None:
                self._bind_tuple_with_rest(pattern, subject_type, rest_index)
            else:
                element_types = self._get_tuple_element_types(subject_type, len(pattern.elements))
                for elem_pattern, elem_type in zip(pattern.elements, element_types):
                    self._bind_pattern_variables(elem_pattern, elem_type)
        elif isinstance(pattern, ConstructorPattern):
            # Handle constructor patterns (Some, Ok, Err, etc.)
            inner_type = self._get_constructor_inner_type(pattern.name, subject_type)
            for arg_pattern in pattern.args:
                self._bind_pattern_variables(arg_pattern, inner_type)
        elif isinstance(pattern, RangePattern):
            # Range patterns don't bind variables
            pass
        elif isinstance(pattern, OrPattern):
            # All alternatives in an or pattern should bind the same variables
            # Bind variables from the first alternative (they should all be the same)
            if pattern.patterns:
                self._bind_pattern_variables(pattern.patterns[0], subject_type)
        elif isinstance(pattern, BindingPattern):
            # Bind the outer name to the full subject type
            self.symbol_table.define(pattern.name, subject_type, pattern.location)
            # Also bind any variables in the inner pattern
            self._bind_pattern_variables(pattern.pattern, subject_type)
        elif isinstance(pattern, RestPattern):
            # Rest patterns bind to a list of the element type
            if pattern.name:
                # Infer element type from list/tuple
                elem_type = self._get_list_element_type(subject_type)
                list_type = GenericTypeInstance("List", (elem_type,))
                self.symbol_table.define(pattern.name, list_type, pattern.location)
        elif isinstance(pattern, ListPattern):
            # For lists, similar to tuples but can have rest patterns
            rest_index = self._find_rest_pattern_index(pattern.elements)
            if rest_index is not None:
                self._bind_list_with_rest(pattern, subject_type, rest_index)
            else:
                elem_type = self._get_list_element_type(subject_type)
                for elem_pattern in pattern.elements:
                    self._bind_pattern_variables(elem_pattern, elem_type)
        elif isinstance(pattern, EnumPattern):
            # Handle enum patterns
            enum_type = self.enum_types.get(pattern.enum_name)
            if enum_type:
                variant = next(
                    (v for v in enum_type.variants if v.name == pattern.variant_name),
                    None
                )
                if variant:
                    for binding, field_type in zip(pattern.bindings, variant.fields):
                        binding_type = self._resolve_type_annotation(field_type)
                        self._bind_pattern_variables(binding, binding_type)

    def _find_rest_pattern_index(
        self, elements: tuple[Pattern, ...]
    ) -> Optional[int]:
        """Find the index of a RestPattern in a sequence of patterns."""
        for i, elem in enumerate(elements):
            if isinstance(elem, RestPattern):
                return i
        return None

    def _bind_tuple_with_rest(
        self, pattern: TuplePattern, subject_type: Type, rest_index: int
    ) -> None:
        """Bind variables in a tuple pattern that contains a rest pattern."""
        element_types = self._get_tuple_element_types(subject_type, -1)  # -1 means variable length
        if not element_types:
            element_types = [ANY_TYPE]

        elements = pattern.elements
        n_elements = len(elements)
        n_before_rest = rest_index
        n_after_rest = n_elements - rest_index - 1

        # Bind elements before rest
        for i in range(n_before_rest):
            elem_type = element_types[i] if i < len(element_types) else ANY_TYPE
            self._bind_pattern_variables(elements[i], elem_type)

        # Bind the rest pattern (to a list/tuple of remaining types)
        rest_pattern = elements[rest_index]
        if isinstance(rest_pattern, RestPattern) and rest_pattern.name:
            # Rest captures remaining elements as a tuple
            rest_type = TupleType(tuple(element_types[n_before_rest:]))
            self.symbol_table.define(rest_pattern.name, rest_type, rest_pattern.location)

        # Bind elements after rest
        for i, elem in enumerate(elements[rest_index + 1:]):
            # These bind from the end of the tuple
            elem_type = element_types[-(n_after_rest - i)] if element_types else ANY_TYPE
            self._bind_pattern_variables(elem, elem_type)

    def _bind_list_with_rest(
        self, pattern: ListPattern, subject_type: Type, rest_index: int
    ) -> None:
        """Bind variables in a list pattern that contains a rest pattern."""
        elem_type = self._get_list_element_type(subject_type)

        elements = pattern.elements

        # Bind all non-rest elements to the element type
        for i, elem in enumerate(elements):
            if isinstance(elem, RestPattern):
                if elem.name:
                    # Rest captures remaining elements as a list
                    list_type = GenericTypeInstance("List", (elem_type,))
                    self.symbol_table.define(elem.name, list_type, elem.location)
            else:
                self._bind_pattern_variables(elem, elem_type)

    def _get_list_element_type(self, list_type: Type) -> Type:
        """Get the element type from a list type."""
        if isinstance(list_type, GenericTypeInstance):
            if list_type.base_name == "List" and list_type.type_args:
                return list_type.type_args[0]
        return ANY_TYPE

    def _get_tuple_element_types(self, tuple_type: Type, count: int) -> list[Type]:
        """Get element types from a tuple type."""
        if isinstance(tuple_type, TupleType):
            return list(tuple_type.element_types)
        if isinstance(tuple_type, GenericTypeInstance) and tuple_type.base_name == "Tuple":
            return list(tuple_type.type_args)
        # If we can't determine types, use Any for each element
        return [ANY_TYPE] * count

    def _get_constructor_inner_type(self, constructor: str, subject_type: Type) -> Type:
        """Get the inner type for a constructor pattern."""
        if isinstance(subject_type, OptionalType):
            if constructor == "Some":
                return subject_type.inner_type
        if isinstance(subject_type, ResultType):
            if constructor == "Ok":
                return subject_type.ok_type
            if constructor == "Err":
                return subject_type.err_type
        if isinstance(subject_type, GenericTypeInstance):
            if constructor in ("Some", "Ok", "Err") and subject_type.type_args:
                return subject_type.type_args[0]
        return ANY_TYPE

    def _is_wildcard_pattern(self, pattern: Pattern) -> bool:
        """Check if a pattern matches all values (wildcard or bare identifier)."""
        if isinstance(pattern, IdentifierPattern):
            return True  # Any identifier pattern matches all
        if isinstance(pattern, BindingPattern):
            # Binding pattern is wildcard if inner pattern is wildcard
            return self._is_wildcard_pattern(pattern.pattern)
        if isinstance(pattern, OrPattern):
            # Or pattern is wildcard if any alternative is wildcard
            return any(self._is_wildcard_pattern(p) for p in pattern.patterns)
        if isinstance(pattern, RestPattern):
            # Rest pattern alone matches all remaining elements
            return True
        return False

    def _check_exhaustiveness(
        self,
        expr: MatchExpression,
        subject_type: Type,
        has_wildcard: bool,
    ) -> None:
        """
        Check that the match expression is exhaustive.

        Reports warnings for non-exhaustive matches.
        """
        if has_wildcard:
            return  # Wildcard covers all cases

        # For Bool type, check that both true and false are covered
        if subject_type == BOOL_TYPE:
            has_true = False
            has_false = False
            for arm in expr.arms:
                if isinstance(arm.pattern, LiteralPattern):
                    if isinstance(arm.pattern.value, BooleanLiteral):
                        if arm.pattern.value.value is True:
                            has_true = True
                        else:
                            has_false = True
            if not (has_true and has_false):
                missing = []
                if not has_true:
                    missing.append("true")
                if not has_false:
                    missing.append("false")
                self._error(
                    f"Non-exhaustive match: missing patterns for {', '.join(missing)}",
                    expr.location,
                )
            return

        # For Optional types, check Some and None
        if isinstance(subject_type, OptionalType) or (
            isinstance(subject_type, GenericTypeInstance) and subject_type.base_name == "Optional"
        ):
            has_some = False
            has_none = False
            for arm in expr.arms:
                if isinstance(arm.pattern, ConstructorPattern):
                    if arm.pattern.name == "Some":
                        has_some = True
                elif isinstance(arm.pattern, LiteralPattern):
                    if isinstance(arm.pattern.value, NoneLiteral):
                        has_none = True
            if not (has_some and has_none):
                missing = []
                if not has_some:
                    missing.append("Some(_)")
                if not has_none:
                    missing.append("None")
                self._error(
                    f"Non-exhaustive match: missing patterns for {', '.join(missing)}",
                    expr.location,
                )
            return

        # For Result types, check Ok and Err
        if isinstance(subject_type, ResultType) or (
            isinstance(subject_type, GenericTypeInstance) and subject_type.base_name == "Result"
        ):
            has_ok = False
            has_err = False
            for arm in expr.arms:
                if isinstance(arm.pattern, ConstructorPattern):
                    if arm.pattern.name == "Ok":
                        has_ok = True
                    elif arm.pattern.name == "Err":
                        has_err = True
            if not (has_ok and has_err):
                missing = []
                if not has_ok:
                    missing.append("Ok(_)")
                if not has_err:
                    missing.append("Err(_)")
                self._error(
                    f"Non-exhaustive match: missing patterns for {', '.join(missing)}",
                    expr.location,
                )
            return

        # For other types (Int, String, etc.), we can only warn without a wildcard
        # Since these types have infinite domains, we cannot check all cases
        # Just warn that a wildcard or identifier pattern should be added
        has_catch_all = any(
            isinstance(arm.pattern, IdentifierPattern) and not arm.guard
            for arm in expr.arms
        )
        if not has_catch_all:
            # Only warn if no guards - guards make exhaustiveness hard to check
            has_guards = any(arm.guard is not None for arm in expr.arms)
            if not has_guards:
                self._error(
                    f"Non-exhaustive match: consider adding a wildcard pattern '_' "
                    f"or catch-all identifier pattern to handle all cases of type {subject_type}",
                    expr.location,
                )

    def _unify_types(
        self,
        types: list[Type],
        location: Optional[SourceLocation],
    ) -> Type:
        """
        Find a common type that can represent all given types.

        This is used for list literals, conditional expressions, etc.
        """
        if not types:
            return UNKNOWN_TYPE

        # Filter out unknown types
        known_types = [t for t in types if not isinstance(t, UnknownType)]
        if not known_types:
            return UNKNOWN_TYPE

        # If all types are the same, return that type
        first = known_types[0]
        if all(t == first for t in known_types):
            return first

        # Int and Float unify to Float
        if all(isinstance(t, PrimitiveType) and t.is_numeric() for t in known_types):
            if any(t == FLOAT_TYPE for t in known_types):
                return FLOAT_TYPE
            return INT_TYPE

        # If any type is Any, result is Any
        if any(isinstance(t, AnyType) for t in known_types):
            return ANY_TYPE

        # For incompatible types, return Any and potentially warn
        return ANY_TYPE

    # -------------------------------------------------------------------------
    # Type Checking for Statements
    # -------------------------------------------------------------------------

    def visit_program(self, node: Program) -> None:
        """Type check the entire program."""
        # First pass: collect function and class signatures
        for stmt in node.statements:
            if isinstance(stmt, FunctionDef):
                self._register_function_signature(stmt)
            elif isinstance(stmt, ClassDef):
                self._register_class_type(stmt)
            elif isinstance(stmt, SceneDef):
                self._register_scene_type(stmt)

        # Second pass: type check all statements
        for stmt in node.statements:
            self.visit(stmt)

    def _register_function_signature(self, func: FunctionDef) -> None:
        """Register a function's signature for later use."""
        param_types: list[Type] = []
        param_names: list[str] = []
        has_defaults: list[bool] = []

        for param in func.parameters:
            param_names.append(param.name)
            if param.type_annotation:
                param_type = self._resolve_type_annotation(param.type_annotation)
            else:
                param_type = ANY_TYPE  # Infer later if possible
            param_types.append(param_type)
            has_defaults.append(param.default_value is not None)

        return_type: Type
        if func.return_type:
            return_type = self._resolve_type_annotation(func.return_type)
        else:
            return_type = NONE_TYPE  # Will be inferred from return statements

        sig = FunctionSignature(
            name=func.name,
            param_types=param_types,
            param_names=param_names,
            return_type=return_type,
            has_defaults=has_defaults,
            location=func.location,
        )

        self.function_signatures[func.name] = sig
        self.symbol_table.define(func.name, sig.to_function_type())

    def _register_class_type(self, cls: ClassDef) -> None:
        """Register a class type for later use."""
        class_type = ClassType(cls.name, cls.base_classes)
        self.class_types[cls.name] = class_type
        self.symbol_table.define(cls.name, class_type)

    def _register_scene_type(self, scene: SceneDef) -> None:
        """Register a scene type (inherits from Scene)."""
        scene_type = ClassType(scene.name, ("Scene",))
        self.class_types[scene.name] = scene_type
        self.symbol_table.define(scene.name, scene_type)

    def visit_let_statement(self, node: LetStatement) -> None:
        """Type check a variable declaration."""
        declared_type: Optional[Type] = None
        if node.type_annotation:
            declared_type = self._resolve_type_annotation(node.type_annotation)

        inferred_type: Optional[Type] = None
        if node.value:
            inferred_type = self._infer_expression_type(node.value)

        # Determine the final type
        if declared_type and inferred_type:
            # Check that the value type is compatible with the declared type
            if not inferred_type.is_compatible_with(declared_type):
                context = f"in assignment to variable '{node.name}'"
                self._error_type_mismatch(declared_type, inferred_type, node.location, context)
            elif inferred_type != declared_type and not isinstance(declared_type, (UnknownType, AnyType)):
                # Record implicit conversion
                if inferred_type == INT_TYPE and declared_type == FLOAT_TYPE:
                    self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, node.location)
            final_type = declared_type
        elif declared_type:
            final_type = declared_type
        elif inferred_type:
            final_type = inferred_type
        else:
            # No type information available
            self._error(
                f"Cannot determine type of variable '{node.name}' without type annotation or initializer",
                node.location,
            )
            final_type = UNKNOWN_TYPE

        # Check for redefinition in same scope
        if self.symbol_table.is_defined_locally(node.name):
            self._error(
                f"Variable '{node.name}' is already defined in this scope",
                node.location,
            )

        # Define with location tracking for better error messages
        self.symbol_table.define(node.name, final_type, node.location)

    def visit_const_declaration(self, node: ConstDeclaration) -> None:
        """
        Type check a compile-time constant declaration.

        Constants are immutable and must have an initializer.
        They can be used as compile-time values.
        """
        # Constants must have a value (enforced by parser, but double-check)
        if node.value is None:
            self._error(
                f"Constant '{node.name}' must have an initializer",
                node.location,
            )
            return

        # Infer the type from the value
        inferred_type = self._infer_expression_type(node.value)

        # Check declared type if present
        declared_type: Optional[Type] = None
        if node.type_annotation:
            declared_type = self._resolve_type_annotation(node.type_annotation)
            if not inferred_type.is_compatible_with(declared_type):
                context = f"in constant declaration '{node.name}'"
                self._error_type_mismatch(declared_type, inferred_type, node.location, context)

        final_type = declared_type if declared_type else inferred_type

        # Check for redefinition in same scope
        if self.symbol_table.is_defined_locally(node.name):
            self._error(
                f"Constant '{node.name}' is already defined in this scope",
                node.location,
            )

        # Define the constant as immutable
        self.symbol_table.define(
            node.name,
            final_type,
            node.location,
            is_mutable=False,  # Mark as immutable
        )

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        """Type check an assignment statement."""
        target_type = self._infer_expression_type(node.target)
        value_type = self._infer_expression_type(node.value)

        if not value_type.is_compatible_with(target_type):
            self._error(
                f"Cannot assign {value_type} to target of type {target_type}",
                node.location,
            )
        elif value_type != target_type and not isinstance(target_type, (UnknownType, AnyType)):
            if value_type == INT_TYPE and target_type == FLOAT_TYPE:
                self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, node.location)

    def visit_compound_assignment(self, node: CompoundAssignment) -> None:
        """Type check a compound assignment (+=, -=, etc.)."""
        target_type = self._infer_expression_type(node.target)
        value_type = self._infer_expression_type(node.value)

        # The result type of the operation
        result_type = self._infer_arithmetic_result_type(
            target_type, value_type, node.operator, node.location
        )

        # Check that result is compatible with target
        if not result_type.is_compatible_with(target_type):
            self._error(
                f"Result type {result_type} is not compatible with target type {target_type}",
                node.location,
            )

    def visit_function_def(self, node: FunctionDef) -> None:
        """Type check a function definition, including generic functions."""
        # Enter type parameter scope if this is a generic function
        if node.type_params:
            self._enter_type_param_scope(node.type_params)

        # Get the registered signature
        sig = self.function_signatures.get(node.name)
        if not sig:
            # Should not happen if visit_program ran first
            self._register_function_signature(node)
            sig = self.function_signatures[node.name]

        # Enter function scope
        self.symbol_table.enter_scope(f"function:{node.name}")

        # Add parameters to scope with location tracking
        for param, param_type in zip(node.parameters, sig.param_types):
            self.symbol_table.define(param.name, param_type, param.location, is_parameter=True)

            # Check default value type
            if param.default_value:
                default_type = self._infer_expression_type(param.default_value)
                if not default_type.is_compatible_with(param_type):
                    context = f"in default value for parameter '{param.name}'"
                    self._error_type_mismatch(param_type, default_type, param.location, context)

        # Set current function return type for checking return statements
        self._current_function_return_type = sig.return_type

        # Type check function body
        if node.body:
            self.visit(node.body)

        self._current_function_return_type = None
        self.symbol_table.exit_scope()

        # Exit type parameter scope
        if node.type_params:
            self._exit_type_param_scope()

    def visit_class_def(self, node: ClassDef) -> None:
        """Type check a class definition."""
        # Enter class scope
        self.symbol_table.enter_scope(f"class:{node.name}")

        # Add 'self' to scope
        class_type = self.class_types.get(node.name, ClassType(node.name))
        self.symbol_table.define("self", class_type)

        # Type check class body
        if node.body:
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    self._register_function_signature(stmt)
            for stmt in node.body.statements:
                self.visit(stmt)

        self.symbol_table.exit_scope()

    def visit_scene_def(self, node: SceneDef) -> None:
        """Type check a scene definition."""
        # Enter scene scope
        self.symbol_table.enter_scope(f"scene:{node.name}")

        # Add 'self' to scope
        scene_type = self.class_types.get(node.name, ClassType(node.name, ("Scene",)))
        self.symbol_table.define("self", scene_type)

        # Type check scene body
        if node.body:
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    self._register_function_signature(stmt)
            for stmt in node.body.statements:
                self.visit(stmt)

        self.symbol_table.exit_scope()

    # -------------------------------------------------------------------------
    # OOP Construct Type Checking
    # -------------------------------------------------------------------------

    def visit_struct_def(self, node: StructDef) -> None:
        """Type check a struct definition, including generic structs."""
        # Enter type parameter scope if this is a generic struct
        type_var_scope: dict[str, TypeVariable] = {}
        if node.type_params:
            type_var_scope = self._enter_type_param_scope(node.type_params)

        # Build field types
        field_types: list[tuple[str, Type]] = []
        for field in node.fields:
            field_type = self._resolve_type_annotation(field.type_annotation)
            field_types.append((field.name, field_type))

        if node.type_params:
            # Register as generic struct type
            type_vars = tuple(type_var_scope.values())
            field_dict = {name: typ for name, typ in field_types}
            generic_struct = GenericStructType(
                name=node.name,
                type_params=type_vars,
                fields=field_dict,
            )
            self.generic_struct_types[node.name] = generic_struct

            # Exit type parameter scope
            self._exit_type_param_scope()
        else:
            # Register as concrete struct type
            struct_type = StructType(name=node.name, fields=tuple(field_types))
            self.struct_types[node.name] = struct_type

            # Add constructor function signature
            param_types = [ft for _, ft in field_types]
            param_names = [fn for fn, _ in field_types]
            self.function_signatures[node.name] = FunctionSignature(
                name=node.name,
                param_types=param_types,
                param_names=param_names,
                return_type=struct_type,
                location=node.location,
            )

    def visit_impl_block(self, node: ImplBlock) -> None:
        """Type check an implementation block, including operator trait implementations."""
        # Get or create the target type
        target_type: Optional[Type] = None
        if node.target_type in self.struct_types:
            target_type = self.struct_types[node.target_type]
        elif node.target_type in self.class_types:
            target_type = self.class_types[node.target_type]
        else:
            self._emit_error(
                f"Unknown type '{node.target_type}' in impl block",
                node.location,
            )
            target_type = ClassType(node.target_type)

        # If this is a trait implementation, check the trait exists (unless it's a built-in operator trait)
        if node.trait_name:
            if node.trait_name not in self.trait_types and not is_operator_trait(node.trait_name):
                self._emit_error(
                    f"Unknown trait '{node.trait_name}'",
                    node.location,
                )

        # Track impl blocks for this type
        if node.target_type not in self.impl_blocks:
            self.impl_blocks[node.target_type] = []
        self.impl_blocks[node.target_type].append(node)

        # Process associated types (e.g., type Output = Vector)
        associated_type_map: dict[str, Type] = {}
        for assoc_type in node.associated_types:
            resolved_type = self._resolve_type_annotation(assoc_type.type_value)
            associated_type_map[assoc_type.name] = resolved_type

        # If this is an operator trait implementation, register it
        if node.trait_name and is_operator_trait(node.trait_name):
            self._register_operator_impl(node, target_type, associated_type_map)

        # Type check each method
        for method in node.methods:
            self._type_check_method(method, target_type, node.target_type)

    def _register_operator_impl(
        self,
        node: ImplBlock,
        target_type: Type,
        associated_types: dict[str, Type],
    ) -> None:
        """
        Register an operator trait implementation for type checking.

        This enables operator overloading for user-defined types.

        Example:
            impl Add for Vector registers (Vector, Add) -> (method_sig, output_type)
            impl Mul<Float> for Vector registers (Vector, Mul<Float>) -> (method_sig, output_type)
        """
        trait_name = node.trait_name
        if not trait_name:
            return

        trait_info = OPERATOR_TRAITS.get(trait_name)
        if not trait_info:
            return

        # Build trait key including type arguments for generic traits
        trait_key = trait_name
        if node.trait_type_args:
            type_args_str = ", ".join(
                str(self._resolve_type_annotation(arg)) for arg in node.trait_type_args
            )
            trait_key = f"{trait_name}<{type_args_str}>"

        # Find the method that implements the operator
        expected_method_name = trait_info.method_name
        impl_method: Optional[Method] = None
        for method in node.methods:
            if method.name == expected_method_name:
                impl_method = method
                break

        if impl_method is None:
            self._emit_error(
                f"Operator trait '{trait_name}' requires method '{expected_method_name}'",
                node.location,
            )
            return

        # Build the method signature
        param_types: list[Type] = []
        for param in impl_method.parameters:
            if param.type_annotation:
                param_types.append(self._resolve_type_annotation(param.type_annotation))
            else:
                param_types.append(UNKNOWN_TYPE)

        return_type = NONE_TYPE
        if impl_method.return_type:
            return_type = self._resolve_type_annotation(impl_method.return_type)

        method_sig = FunctionType(
            param_types=tuple(param_types),
            return_type=return_type,
        )

        # Get the output type (from associated types or return type)
        output_type: Optional[Type] = associated_types.get("Output", return_type)

        # For comparison operators, output type is always Bool
        if trait_info.kind == OperatorKind.COMPARISON:
            output_type = BOOL_TYPE

        # Register the operator implementation
        self.operator_impls[(node.target_type, trait_key)] = (method_sig, output_type)

        # Track trait implementation for this type
        if node.target_type not in self.trait_implementations:
            self.trait_implementations[node.target_type] = set()
        self.trait_implementations[node.target_type].add(trait_name)

    def _type_check_method(
        self, method: Method, target_type: Type, type_name: str
    ) -> None:
        """Type check a method definition."""
        # Enter method scope
        self.symbol_table.enter_scope(f"method:{type_name}.{method.name}")

        # Add 'self' if this is an instance method
        if method.has_self:
            self.symbol_table.define("self", target_type)

        # Build parameter types
        param_types: list[Type] = []
        param_names: list[str] = []
        for param in method.parameters:
            param_type = UNKNOWN_TYPE
            if param.type_annotation:
                param_type = self._resolve_type_annotation(param.type_annotation)
            self.symbol_table.define(param.name, param_type, param.location, is_parameter=True)
            param_types.append(param_type)
            param_names.append(param.name)

        # Determine return type
        return_type = NONE_TYPE
        if method.return_type:
            return_type = self._resolve_type_annotation(method.return_type)

        # Set current function return type for return statement checking
        prev_return_type = self._current_function_return_type
        self._current_function_return_type = return_type

        # Register the method signature
        full_method_name = f"{type_name}.{method.name}"
        self.function_signatures[full_method_name] = FunctionSignature(
            name=method.name,
            param_types=param_types,
            param_names=param_names,
            return_type=return_type,
            location=method.location,
        )

        # Type check method body
        if method.body:
            self.visit(method.body)

        self._current_function_return_type = prev_return_type
        self.symbol_table.exit_scope()

    def visit_trait_def(self, node: TraitDef) -> None:
        """Type check a trait definition."""
        # Build method signatures
        method_sigs: list[tuple[str, FunctionType]] = []
        for method in node.methods:
            # Build parameter types
            param_types: list[Type] = []
            for param in method.parameters:
                if param.type_annotation:
                    param_types.append(self._resolve_type_annotation(param.type_annotation))
                else:
                    param_types.append(ANY_TYPE)

            return_type = NONE_TYPE
            if method.return_type:
                return_type = self._resolve_type_annotation(method.return_type)

            func_type = FunctionType(
                param_types=tuple(param_types),
                return_type=return_type,
            )
            method_sigs.append((method.name, func_type))

        # Register trait type
        trait_type = TraitType(name=node.name, methods=tuple(method_sigs))
        self.trait_types[node.name] = trait_type

    def visit_enum_def(self, node: EnumDef) -> None:
        """Type check an enum definition."""
        # Build variant types
        variants: list[tuple[str, tuple[Type, ...]]] = []
        for variant in node.variants:
            field_types: list[Type] = []
            for field_type_ann in variant.fields:
                field_types.append(self._resolve_type_annotation(field_type_ann))
            variants.append((variant.name, tuple(field_types)))

        # Register enum type
        enum_type = EnumType(name=node.name, variants=tuple(variants))
        self.enum_types[node.name] = enum_type

        # Register constructor functions for each variant
        for variant_name, field_types in variants:
            full_name = f"{node.name}::{variant_name}"
            if field_types:
                # Variant with associated data acts as constructor
                self.function_signatures[full_name] = FunctionSignature(
                    name=variant_name,
                    param_types=list(field_types),
                    param_names=[f"_{i}" for i in range(len(field_types))],
                    return_type=enum_type,
                    location=node.location,
                )

    def visit_self_expression(self, node: SelfExpression) -> Type:
        """Type check a self expression."""
        self_type = self.symbol_table.lookup("self")
        if self_type is None:
            self._emit_error("'self' is only valid inside a method", node.location)
            return UNKNOWN_TYPE
        return self_type

    def visit_enum_variant_access(self, node: EnumVariantAccess) -> Type:
        """Type check an enum variant access."""
        if node.enum_name not in self.enum_types:
            self._emit_error(f"Unknown enum type '{node.enum_name}'", node.location)
            return UNKNOWN_TYPE

        enum_type = self.enum_types[node.enum_name]
        variant_types = enum_type.get_variant(node.variant_name)

        if variant_types is None:
            self._emit_error(
                f"Unknown variant '{node.variant_name}' in enum '{node.enum_name}'",
                node.location,
            )
            return UNKNOWN_TYPE

        if variant_types:
            # Variant with associated data - return constructor type
            return EnumVariantType(
                enum_type=enum_type,
                variant_name=node.variant_name,
                associated_types=variant_types,
            )
        else:
            # Variant without data - return the enum type directly
            return enum_type

    def visit_struct_literal(self, node: StructLiteral) -> Type:
        """Type check a struct literal."""
        if node.struct_name not in self.struct_types:
            self._emit_error(
                f"Unknown struct type '{node.struct_name}'", node.location
            )
            return UNKNOWN_TYPE

        struct_type = self.struct_types[node.struct_name]

        # Check that all required fields are provided
        provided_fields = {name for name, _ in node.fields}
        expected_fields = {name for name, _ in struct_type.fields}

        missing = expected_fields - provided_fields
        extra = provided_fields - expected_fields

        if missing:
            self._emit_error(
                f"Missing fields in struct literal: {', '.join(sorted(missing))}",
                node.location,
            )

        if extra:
            self._emit_error(
                f"Unknown fields in struct literal: {', '.join(sorted(extra))}",
                node.location,
            )

        # Type check each field value
        for field_name, field_value in node.fields:
            expected_type = struct_type.get_field_type(field_name)
            if expected_type:
                actual_type = self._infer_expression_type(field_value)
                if not actual_type.is_compatible_with(expected_type):
                    self._emit_error(
                        f"Type mismatch for field '{field_name}': expected {expected_type}, got {actual_type}",
                        field_value.location,
                    )

        return struct_type

    def visit_enum_pattern(self, node: EnumPattern) -> None:
        """Type check an enum pattern."""
        if node.enum_name not in self.enum_types:
            self._emit_error(
                f"Unknown enum type '{node.enum_name}'", node.location
            )
            return

        enum_type = self.enum_types[node.enum_name]
        variant_types = enum_type.get_variant(node.variant_name)

        if variant_types is None:
            self._emit_error(
                f"Unknown variant '{node.variant_name}' in enum '{node.enum_name}'",
                node.location,
            )
            return

        # Check binding count matches variant field count
        if len(node.bindings) != len(variant_types):
            self._emit_error(
                f"Wrong number of bindings for variant '{node.variant_name}': "
                f"expected {len(variant_types)}, got {len(node.bindings)}",
                node.location,
            )

    def visit_if_statement(self, node: IfStatement) -> None:
        """Type check an if statement."""
        cond_type = self._infer_expression_type(node.condition)
        self._check_boolean_operand(cond_type, "if condition", node.condition.location)

        # Type check then block
        self.symbol_table.enter_scope("if:then")
        self.visit(node.then_block)
        self.symbol_table.exit_scope()

        # Type check elif clauses
        for elif_cond, elif_block in node.elif_clauses:
            elif_cond_type = self._infer_expression_type(elif_cond)
            self._check_boolean_operand(elif_cond_type, "elif condition", elif_cond.location)

            self.symbol_table.enter_scope("if:elif")
            self.visit(elif_block)
            self.symbol_table.exit_scope()

        # Type check else block
        if node.else_block:
            self.symbol_table.enter_scope("if:else")
            self.visit(node.else_block)
            self.symbol_table.exit_scope()

    def visit_for_statement(self, node: ForStatement) -> None:
        """Type check a for statement."""
        iterable_type = self._infer_expression_type(node.iterable)

        # Determine the element type from the iterable
        element_type = self._get_iterable_element_type(iterable_type, node.iterable.location)

        # Enter loop scope
        self.symbol_table.enter_scope("for")
        self._in_loop = True

        # Add loop variable(s) to scope
        if isinstance(node.variable, tuple):
            # Tuple destructuring: for (a, b) in ...
            # Each variable gets ANY_TYPE since we can't infer tuple element types easily
            for var in node.variable:
                self.symbol_table.define(var, ANY_TYPE)
        else:
            self.symbol_table.define(node.variable, element_type)

        # Type check loop body
        self.visit(node.body)

        self._in_loop = False
        self.symbol_table.exit_scope()

    def _get_iterable_element_type(
        self,
        iterable_type: Type,
        location: Optional[SourceLocation],
    ) -> Type:
        """Determine the element type of an iterable."""
        if isinstance(iterable_type, GenericTypeInstance):
            if iterable_type.base_name in ("List", "Set"):
                return iterable_type.type_args[0] if iterable_type.type_args else ANY_TYPE
            if iterable_type.base_name == "Dict":
                # Iterating over dict yields keys
                return iterable_type.type_args[0] if iterable_type.type_args else ANY_TYPE

        if isinstance(iterable_type, ArrayType):
            return iterable_type.element_type or FLOAT_TYPE

        if isinstance(iterable_type, RangeType):
            return iterable_type.element_type

        if iterable_type == STRING_TYPE:
            return STRING_TYPE  # Iterating over string yields characters

        if isinstance(iterable_type, (UnknownType, AnyType)):
            return ANY_TYPE

        self._error(f"Cannot iterate over non-iterable type: {iterable_type}", location)
        return ANY_TYPE

    def visit_while_statement(self, node: WhileStatement) -> None:
        """Type check a while statement."""
        cond_type = self._infer_expression_type(node.condition)
        self._check_boolean_operand(cond_type, "while condition", node.condition.location)

        # Enter loop scope
        self.symbol_table.enter_scope("while")
        self._in_loop = True

        self.visit(node.body)

        self._in_loop = False
        self.symbol_table.exit_scope()

    def visit_if_let_statement(self, node: "IfLetStatement") -> None:
        """Type check an if let statement."""
        from mathviz.compiler.ast_nodes import IfLetStatement

        # Infer the type of the value expression
        self._infer_expression_type(node.value)

        # Enter scope for the then block (pattern bindings are visible here)
        self.symbol_table.enter_scope("if_let")

        # Define bindings from pattern as ANY_TYPE (we don't have full pattern typing yet)
        self._define_pattern_bindings(node.pattern)

        self.visit(node.then_block)
        self.symbol_table.exit_scope()

        if node.else_block:
            self.symbol_table.enter_scope("if_let_else")
            self.visit(node.else_block)
            self.symbol_table.exit_scope()

    def visit_while_let_statement(self, node: "WhileLetStatement") -> None:
        """Type check a while let statement."""
        from mathviz.compiler.ast_nodes import WhileLetStatement

        # Infer the type of the value expression
        self._infer_expression_type(node.value)

        # Enter loop scope
        self.symbol_table.enter_scope("while_let")
        self._in_loop = True

        # Define bindings from pattern as ANY_TYPE
        self._define_pattern_bindings(node.pattern)

        self.visit(node.body)

        self._in_loop = False
        self.symbol_table.exit_scope()

    def _define_pattern_bindings(self, pattern: "Pattern") -> None:
        """Define variables from a pattern in the current scope."""
        from mathviz.compiler.ast_nodes import (
            IdentifierPattern, TuplePattern, ConstructorPattern, BindingPattern
        )

        if isinstance(pattern, IdentifierPattern):
            if pattern.name != "_":
                self.symbol_table.define(pattern.name, ANY_TYPE)
        elif isinstance(pattern, TuplePattern):
            for elem in pattern.elements:
                self._define_pattern_bindings(elem)
        elif isinstance(pattern, ConstructorPattern):
            for arg in pattern.arguments:
                self._define_pattern_bindings(arg)
        elif isinstance(pattern, BindingPattern):
            self.symbol_table.define(pattern.name, ANY_TYPE)
            self._define_pattern_bindings(pattern.pattern)

    def visit_return_statement(self, node: ReturnStatement) -> None:
        """Type check a return statement."""
        if self._current_function_return_type is None:
            span = self._location_to_span(node.location, 6)  # len("return")
            if span and self.source:
                create_return_outside_function_diagnostic(self._get_emitter(), span)
            self._error("'return' outside of function", node.location)
            return

        if node.value:
            value_type = self._infer_expression_type(node.value)
            expected_type = self._current_function_return_type

            if not value_type.is_compatible_with(expected_type):
                self._error(
                    f"Return type {value_type} does not match declared return type {expected_type}",
                    node.location,
                )
            elif value_type != expected_type and not isinstance(expected_type, (UnknownType, AnyType)):
                if value_type == INT_TYPE and expected_type == FLOAT_TYPE:
                    self._record_conversion(INT_TYPE, FLOAT_TYPE, TypeConversion.INT_TO_FLOAT, node.location)
        else:
            # Return without value
            if self._current_function_return_type != NONE_TYPE:
                if not isinstance(self._current_function_return_type, (UnknownType, AnyType)):
                    self._error(
                        f"Function expects return type {self._current_function_return_type}, "
                        f"but return has no value",
                        node.location,
                    )

    def visit_break_statement(self, node: BreakStatement) -> None:
        """Type check a break statement."""
        if not self._in_loop:
            span = self._location_to_span(node.location, 5)  # len("break")
            if span and self.source:
                create_break_outside_loop_diagnostic(self._get_emitter(), span)
            self._error("'break' outside of loop", node.location)

    def visit_continue_statement(self, node: ContinueStatement) -> None:
        """Type check a continue statement."""
        if not self._in_loop:
            span = self._location_to_span(node.location, 8)  # len("continue")
            if span and self.source:
                self._get_emitter().error(
                    ErrorCode.E0303,
                    "'continue' outside of loop",
                    span,
                ).help("'continue' can only be used inside 'for' or 'while' loops").emit()
            self._error("'continue' outside of loop", node.location)

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        """Type check an expression statement."""
        self._infer_expression_type(node.expression)

    def visit_print_statement(self, node: PrintStatement) -> None:
        """Type check a print statement."""
        format_type = self._infer_expression_type(node.format_string)
        if format_type != STRING_TYPE and not isinstance(format_type, (UnknownType, AnyType)):
            self._error(
                f"Print format string must be String, got {format_type}",
                node.format_string.location,
            )

        # Type check format arguments
        for arg in node.arguments:
            self._infer_expression_type(arg)

    def visit_play_statement(self, node: PlayStatement) -> None:
        """Type check a Manim play statement."""
        anim_type = self._infer_expression_type(node.animation)

        # Animation should be an Animation type
        expected = ClassType("Animation")
        if not anim_type.is_compatible_with(expected) and not isinstance(anim_type, (UnknownType, AnyType)):
            # Only warn, as Manim has many animation types
            pass

        if node.run_time:
            run_time_type = self._infer_expression_type(node.run_time)
            if not self._is_numeric_type(run_time_type):
                self._error(
                    f"run_time must be numeric, got {run_time_type}",
                    node.run_time.location,
                )

    def visit_wait_statement(self, node: WaitStatement) -> None:
        """Type check a Manim wait statement."""
        if node.duration:
            duration_type = self._infer_expression_type(node.duration)
            if not self._is_numeric_type(duration_type):
                self._error(
                    f"wait duration must be numeric, got {duration_type}",
                    node.duration.location,
                )

    def visit_import_statement(self, node: ImportStatement) -> None:
        """Handle import statements."""
        # Import statements add module names to the symbol table
        if node.alias:
            self.symbol_table.define(node.alias, ANY_TYPE)
        elif node.is_from_import:
            for name, alias in node.names:
                self.symbol_table.define(alias or name, ANY_TYPE)
        else:
            self.symbol_table.define(node.module, ANY_TYPE)

    def visit_use_statement(self, node: UseStatement) -> None:
        """Handle use statements."""
        # Use statements import module contents
        if node.alias:
            self.symbol_table.define(node.alias, ANY_TYPE)
        elif node.wildcard:
            # Wildcard imports add all exports (we don't track these)
            pass
        else:
            # Import the module path
            module_name = ".".join(node.module_path)
            self.symbol_table.define(node.module_path[-1], ANY_TYPE)

    def visit_module_decl(self, node: ModuleDecl) -> None:
        """Type check a module declaration."""
        # Enter module scope
        self.symbol_table.enter_scope(f"module:{node.name}")

        # Type check module body
        self.visit(node.body)

        self.symbol_table.exit_scope()

    # -------------------------------------------------------------------------
    # Type Annotation Visitors (no-op, just for completeness)
    # -------------------------------------------------------------------------

    def visit_simple_type(self, node: SimpleType) -> None:
        """Visit a simple type annotation."""
        pass

    def visit_generic_type(self, node: GenericType) -> None:
        """Visit a generic type annotation."""
        pass

    def visit_function_type(self, node: ASTFunctionType) -> None:
        """Visit a function type annotation."""
        pass


# =============================================================================
# Utility Functions
# =============================================================================


def type_check(program: Program) -> list[MathVizTypeError]:
    """
    Convenience function to type check a program.

    Args:
        program: The Program AST to check

    Returns:
        A list of type errors found
    """
    checker = TypeChecker()
    return checker.check(program)


def infer_expression_type(
    expr: Expression,
    context: Optional[dict[str, Type]] = None,
) -> Type:
    """
    Infer the type of an expression with an optional context.

    Args:
        expr: The expression to infer
        context: Optional variable bindings for type inference

    Returns:
        The inferred Type
    """
    checker = TypeChecker()
    if context:
        for name, type_ in context.items():
            checker.symbol_table.define(name, type_)
    return checker.infer_type(expr)


__all__ = [
    # Type classes
    "Type",
    "PrimitiveType",
    "GenericTypeInstance",
    "ArrayType",
    "FunctionType",
    "ClassType",
    "UnknownType",
    "AnyType",
    "RangeType",
    # Type constants
    "INT_TYPE",
    "FLOAT_TYPE",
    "BOOL_TYPE",
    "STRING_TYPE",
    "NONE_TYPE",
    "VEC_TYPE",
    "MAT_TYPE",
    "ARRAY_TYPE",
    "RANGE_TYPE",
    "UNKNOWN_TYPE",
    "ANY_TYPE",
    # Type checking
    "TypeChecker",
    "FunctionSignature",
    "SymbolTable",
    "Scope",
    "TypeConversion",
    "ConversionInfo",
    # Utility functions
    "type_check",
    "infer_expression_type",
]
