"""
Operator traits and overloading support for MathViz.

This module defines the built-in operator traits that can be implemented
to provide custom behavior for operators on user-defined types.

Each operator trait maps to:
- A method name (e.g., "add" for Add)
- The corresponding BinaryOperator or UnaryOperator enum value
- The Python magic method name (e.g., "__add__")
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from mathviz.compiler.ast_nodes import BinaryOperator, UnaryOperator


class OperatorKind(Enum):
    """Classification of operator types."""
    BINARY = auto()
    UNARY = auto()
    COMPARISON = auto()
    INDEX = auto()


@dataclass(frozen=True)
class OperatorTraitInfo:
    """
    Information about an operator trait.

    Attributes:
        method_name: The method name in MathViz (e.g., "add")
        operator: The BinaryOperator or UnaryOperator enum value
        python_magic: The Python magic method name (e.g., "__add__")
        kind: The kind of operator (binary, unary, comparison, index)
        has_output_type: Whether the trait has an associated Output type
        right_operand_generic: Whether the right operand can be generic (e.g., Mul<Float>)
    """
    method_name: str
    operator: Optional[BinaryOperator | UnaryOperator]
    python_magic: str
    kind: OperatorKind
    has_output_type: bool = True
    right_operand_generic: bool = False


# Mapping of trait names to their operator information
OPERATOR_TRAITS: dict[str, OperatorTraitInfo] = {
    # Arithmetic operators (binary, have Output type)
    "Add": OperatorTraitInfo(
        method_name="add",
        operator=BinaryOperator.ADD,
        python_magic="__add__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "Sub": OperatorTraitInfo(
        method_name="sub",
        operator=BinaryOperator.SUB,
        python_magic="__sub__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "Mul": OperatorTraitInfo(
        method_name="mul",
        operator=BinaryOperator.MUL,
        python_magic="__mul__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "Div": OperatorTraitInfo(
        method_name="div",
        operator=BinaryOperator.DIV,
        python_magic="__truediv__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "Mod": OperatorTraitInfo(
        method_name="mod",
        operator=BinaryOperator.MOD,
        python_magic="__mod__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "Pow": OperatorTraitInfo(
        method_name="pow",
        operator=BinaryOperator.POW,
        python_magic="__pow__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "FloorDiv": OperatorTraitInfo(
        method_name="floor_div",
        operator=BinaryOperator.FLOOR_DIV,
        python_magic="__floordiv__",
        kind=OperatorKind.BINARY,
        has_output_type=True,
        right_operand_generic=True,
    ),

    # Comparison operators (return Bool, no Output type)
    "Eq": OperatorTraitInfo(
        method_name="eq",
        operator=BinaryOperator.EQ,
        python_magic="__eq__",
        kind=OperatorKind.COMPARISON,
        has_output_type=False,
        right_operand_generic=True,
    ),
    "Ne": OperatorTraitInfo(
        method_name="ne",
        operator=BinaryOperator.NE,
        python_magic="__ne__",
        kind=OperatorKind.COMPARISON,
        has_output_type=False,
        right_operand_generic=True,
    ),
    "Lt": OperatorTraitInfo(
        method_name="lt",
        operator=BinaryOperator.LT,
        python_magic="__lt__",
        kind=OperatorKind.COMPARISON,
        has_output_type=False,
        right_operand_generic=True,
    ),
    "Le": OperatorTraitInfo(
        method_name="le",
        operator=BinaryOperator.LE,
        python_magic="__le__",
        kind=OperatorKind.COMPARISON,
        has_output_type=False,
        right_operand_generic=True,
    ),
    "Gt": OperatorTraitInfo(
        method_name="gt",
        operator=BinaryOperator.GT,
        python_magic="__gt__",
        kind=OperatorKind.COMPARISON,
        has_output_type=False,
        right_operand_generic=True,
    ),
    "Ge": OperatorTraitInfo(
        method_name="ge",
        operator=BinaryOperator.GE,
        python_magic="__ge__",
        kind=OperatorKind.COMPARISON,
        has_output_type=False,
        right_operand_generic=True,
    ),

    # Unary operators
    "Neg": OperatorTraitInfo(
        method_name="neg",
        operator=UnaryOperator.NEG,
        python_magic="__neg__",
        kind=OperatorKind.UNARY,
        has_output_type=True,
        right_operand_generic=False,
    ),
    "Not": OperatorTraitInfo(
        method_name="not_",
        operator=UnaryOperator.NOT,
        python_magic="__invert__",
        kind=OperatorKind.UNARY,
        has_output_type=True,
        right_operand_generic=False,
    ),
    "Pos": OperatorTraitInfo(
        method_name="pos",
        operator=UnaryOperator.POS,
        python_magic="__pos__",
        kind=OperatorKind.UNARY,
        has_output_type=True,
        right_operand_generic=False,
    ),

    # Index operators
    "Index": OperatorTraitInfo(
        method_name="index",
        operator=None,
        python_magic="__getitem__",
        kind=OperatorKind.INDEX,
        has_output_type=True,
        right_operand_generic=True,
    ),
    "IndexMut": OperatorTraitInfo(
        method_name="index_mut",
        operator=None,
        python_magic="__setitem__",
        kind=OperatorKind.INDEX,
        has_output_type=False,
        right_operand_generic=True,
    ),
}


def get_trait_for_binary_operator(op: BinaryOperator) -> Optional[str]:
    """Get the trait name for a binary operator."""
    for trait_name, info in OPERATOR_TRAITS.items():
        if info.operator == op and info.kind in (OperatorKind.BINARY, OperatorKind.COMPARISON):
            return trait_name
    return None


def get_trait_for_unary_operator(op: UnaryOperator) -> Optional[str]:
    """Get the trait name for a unary operator."""
    for trait_name, info in OPERATOR_TRAITS.items():
        if info.operator == op and info.kind == OperatorKind.UNARY:
            return trait_name
    return None


def is_operator_trait(trait_name: str) -> bool:
    """Check if a trait name is an operator trait."""
    return trait_name in OPERATOR_TRAITS


def get_python_magic_method(trait_name: str) -> Optional[str]:
    """Get the Python magic method name for an operator trait."""
    info = OPERATOR_TRAITS.get(trait_name)
    return info.python_magic if info else None


def get_operator_method_name(trait_name: str) -> Optional[str]:
    """Get the MathViz method name for an operator trait."""
    info = OPERATOR_TRAITS.get(trait_name)
    return info.method_name if info else None
