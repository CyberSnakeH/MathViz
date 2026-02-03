"""
Compile-time Constant Evaluator for MathViz.

This module provides compile-time evaluation of constant expressions,
enabling optimizations and constant folding during compilation.

Features:
- Evaluates arithmetic, logical, and comparison expressions
- Supports built-in mathematical constants (PI, E, TAU, etc.)
- Handles user-defined constants with dependency resolution
- Validates that expressions are constant-evaluable
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from mathviz.compiler.ast_nodes import (
    Expression,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BooleanLiteral,
    NoneLiteral,
    Identifier,
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    MemberAccess,
    ListLiteral,
    TupleLiteral,
    BinaryOperator,
    UnaryOperator,
    ConstDeclaration,
)
from mathviz.utils.errors import SourceLocation


class ConstEvalError(Exception):
    """Raised when an expression cannot be evaluated at compile time."""

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
    ) -> None:
        self.message = message
        self.location = location
        super().__init__(message)


# Built-in mathematical constants
BUILTIN_CONSTANTS: dict[str, Union[int, float, bool, str]] = {
    # Mathematical constants
    "PI": 3.14159265358979323846,
    "E": 2.71828182845904523536,
    "TAU": 6.28318530717958647692,  # 2 * PI
    "PHI": 1.61803398874989484820,  # Golden ratio
    "SQRT2": 1.41421356237309504880,  # sqrt(2)
    "SQRT3": 1.73205080756887729352,  # sqrt(3)
    "LN2": 0.69314718055994530942,  # ln(2)
    "LN10": 2.30258509299404568402,  # ln(10)
    "LOG2E": 1.44269504088896340736,  # log2(e)
    "LOG10E": 0.43429448190325182765,  # log10(e)
    # Special values
    "INF": float("inf"),
    "NEG_INF": float("-inf"),
    "NAN": float("nan"),
    # Boolean constants
    "TRUE": True,
    "FALSE": False,
}


# Built-in functions that can be evaluated at compile time
CONST_EVAL_FUNCTIONS: dict[str, callable] = {
    # Math functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "abs": abs,
    "pow": pow,
    "min": min,
    "max": max,
    # Type conversions
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    # String functions
    "len": len,
}


@dataclass
class ConstantInfo:
    """Information about a user-defined constant."""

    name: str
    value: Any
    type_name: Optional[str] = None
    location: Optional[SourceLocation] = None


class ConstEvaluator:
    """
    Evaluates constant expressions at compile time.

    This class handles:
    - Literal values (integers, floats, strings, booleans)
    - Built-in mathematical constants (PI, E, TAU, etc.)
    - User-defined constants
    - Arithmetic, logical, and comparison operators
    - Compile-time function calls (sqrt, sin, cos, etc.)

    Usage:
        evaluator = ConstEvaluator()
        value = evaluator.evaluate(expr)
        is_const = evaluator.is_const_expr(expr)
    """

    def __init__(self) -> None:
        """Initialize the constant evaluator with built-in constants."""
        self.constants: dict[str, Any] = dict(BUILTIN_CONSTANTS)
        self._evaluating: set[str] = set()  # Track circular dependencies

    def add_constant(
        self,
        name: str,
        value: Any,
        location: Optional[SourceLocation] = None,
    ) -> None:
        """
        Register a user-defined constant.

        Args:
            name: The constant name
            value: The constant value
            location: Source location for error reporting
        """
        if name in self.constants:
            raise ConstEvalError(
                f"Constant '{name}' is already defined",
                location,
            )
        self.constants[name] = value

    def evaluate(self, expr: Expression) -> Any:
        """
        Evaluate a constant expression at compile time.

        Args:
            expr: The expression to evaluate

        Returns:
            The computed value

        Raises:
            ConstEvalError: If the expression cannot be evaluated at compile time
        """
        if isinstance(expr, IntegerLiteral):
            return expr.value

        if isinstance(expr, FloatLiteral):
            return expr.value

        if isinstance(expr, StringLiteral):
            return expr.value

        if isinstance(expr, BooleanLiteral):
            return expr.value

        if isinstance(expr, NoneLiteral):
            return None

        if isinstance(expr, Identifier):
            return self._evaluate_identifier(expr)

        if isinstance(expr, BinaryExpression):
            return self._evaluate_binary(expr)

        if isinstance(expr, UnaryExpression):
            return self._evaluate_unary(expr)

        if isinstance(expr, CallExpression):
            return self._evaluate_call(expr)

        if isinstance(expr, ListLiteral):
            return [self.evaluate(elem) for elem in expr.elements]

        if isinstance(expr, TupleLiteral):
            return tuple(self.evaluate(elem) for elem in expr.elements)

        raise ConstEvalError(
            f"Expression of type {type(expr).__name__} cannot be evaluated at compile time",
            getattr(expr, "location", None),
        )

    def _evaluate_identifier(self, expr: Identifier) -> Any:
        """Evaluate an identifier (constant reference)."""
        name = expr.name

        # Check for circular dependencies
        if name in self._evaluating:
            raise ConstEvalError(
                f"Circular dependency in constant '{name}'",
                expr.location,
            )

        # Check if it's a known constant
        if name not in self.constants:
            raise ConstEvalError(
                f"Unknown constant: '{name}'",
                expr.location,
            )

        return self.constants[name]

    def _evaluate_binary(self, expr: BinaryExpression) -> Any:
        """Evaluate a binary expression."""
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)

        op = expr.operator

        # Arithmetic operators
        if op == BinaryOperator.ADD:
            return left + right
        if op == BinaryOperator.SUB:
            return left - right
        if op == BinaryOperator.MUL:
            return left * right
        if op == BinaryOperator.DIV:
            if right == 0:
                raise ConstEvalError("Division by zero", expr.location)
            return left / right
        if op == BinaryOperator.FLOOR_DIV:
            if right == 0:
                raise ConstEvalError("Division by zero", expr.location)
            return left // right
        if op == BinaryOperator.MOD:
            if right == 0:
                raise ConstEvalError("Modulo by zero", expr.location)
            return left % right
        if op == BinaryOperator.POW:
            return left**right

        # Comparison operators
        if op == BinaryOperator.EQ:
            return left == right
        if op == BinaryOperator.NE:
            return left != right
        if op == BinaryOperator.LT:
            return left < right
        if op == BinaryOperator.GT:
            return left > right
        if op == BinaryOperator.LE:
            return left <= right
        if op == BinaryOperator.GE:
            return left >= right

        # Logical operators
        if op == BinaryOperator.AND:
            return left and right
        if op == BinaryOperator.OR:
            return left or right

        raise ConstEvalError(
            f"Operator {op} cannot be evaluated at compile time",
            expr.location,
        )

    def _evaluate_unary(self, expr: UnaryExpression) -> Any:
        """Evaluate a unary expression."""
        operand = self.evaluate(expr.operand)

        if expr.operator == UnaryOperator.NEG:
            return -operand
        if expr.operator == UnaryOperator.POS:
            return +operand
        if expr.operator == UnaryOperator.NOT:
            return not operand

        raise ConstEvalError(
            f"Unary operator {expr.operator} cannot be evaluated at compile time",
            expr.location,
        )

    def _evaluate_call(self, expr: CallExpression) -> Any:
        """Evaluate a function call at compile time."""
        # Only support simple function calls (no method calls)
        if not isinstance(expr.callee, Identifier):
            raise ConstEvalError(
                "Only simple function calls can be evaluated at compile time",
                expr.location,
            )

        func_name = expr.callee.name

        if func_name not in CONST_EVAL_FUNCTIONS:
            raise ConstEvalError(
                f"Function '{func_name}' cannot be evaluated at compile time",
                expr.location,
            )

        # Evaluate arguments
        args = [self.evaluate(arg) for arg in expr.arguments]

        try:
            return CONST_EVAL_FUNCTIONS[func_name](*args)
        except Exception as e:
            raise ConstEvalError(
                f"Error evaluating {func_name}({args}): {e}",
                expr.location,
            )

    def is_const_expr(self, expr: Expression) -> bool:
        """
        Check if an expression can be evaluated at compile time.

        Args:
            expr: The expression to check

        Returns:
            True if the expression is constant-evaluable
        """
        try:
            self.evaluate(expr)
            return True
        except ConstEvalError:
            return False

    def evaluate_declaration(self, decl: ConstDeclaration) -> Any:
        """
        Evaluate a constant declaration and register the constant.

        Args:
            decl: The constant declaration to evaluate

        Returns:
            The computed value

        Raises:
            ConstEvalError: If the expression cannot be evaluated
        """
        # Check for circular dependencies
        if decl.name in self._evaluating:
            raise ConstEvalError(
                f"Circular dependency in constant '{decl.name}'",
                decl.location,
            )

        self._evaluating.add(decl.name)
        try:
            value = self.evaluate(decl.value)
            self.constants[decl.name] = value
            return value
        finally:
            self._evaluating.discard(decl.name)

    def get_constant(self, name: str) -> Optional[Any]:
        """
        Get the value of a constant by name.

        Args:
            name: The constant name

        Returns:
            The constant value, or None if not found
        """
        return self.constants.get(name)

    def get_all_constants(self) -> dict[str, Any]:
        """
        Get all registered constants.

        Returns:
            A dictionary mapping constant names to their values
        """
        return dict(self.constants)

    def get_user_constants(self) -> dict[str, Any]:
        """
        Get only user-defined constants (excluding built-ins).

        Returns:
            A dictionary of user-defined constants
        """
        return {
            name: value for name, value in self.constants.items() if name not in BUILTIN_CONSTANTS
        }


def evaluate_const(expr: Expression) -> Any:
    """
    Convenience function to evaluate a constant expression.

    Args:
        expr: The expression to evaluate

    Returns:
        The computed value
    """
    return ConstEvaluator().evaluate(expr)


def is_const_expr(expr: Expression) -> bool:
    """
    Convenience function to check if an expression is constant.

    Args:
        expr: The expression to check

    Returns:
        True if the expression can be evaluated at compile time
    """
    return ConstEvaluator().is_const_expr(expr)
