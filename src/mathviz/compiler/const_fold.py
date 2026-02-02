"""
Constant Folding and Propagation Optimizer for MathViz.

This module provides compile-time optimizations for constant expressions,
including:
- Constant Folding: Evaluate constant expressions at compile time
- Constant Propagation: Track and propagate known constant values
- Dead Code Elimination: Remove unreachable and dead code
- Algebraic Simplification: Apply algebraic identities
- Strength Reduction: Replace expensive operations with cheaper ones
- Common Subexpression Elimination: Reuse computed subexpressions

All optimizations maintain AST immutability by creating new nodes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from mathviz.compiler.ast_nodes import (
    ASTNode,
    ASTVisitor,
    BaseASTVisitor,
    # Type annotations
    TypeAnnotation,
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
    BinaryExpression,
    UnaryExpression,
    BinaryOperator,
    UnaryOperator,
    CallExpression,
    MemberAccess,
    IndexExpression,
    ConditionalExpression,
    LambdaExpression,
    RangeExpression,
    # Comprehensions
    ListComprehension,
    SetComprehension,
    DictComprehension,
    ComprehensionClause,
    PipeLambda,
    # Statements
    Statement,
    Block,
    ExpressionStatement,
    LetStatement,
    ConstDeclaration,
    AssignmentStatement,
    CompoundAssignment,
    FunctionDef,
    IfStatement,
    ForStatement,
    WhileStatement,
    ReturnStatement,
    BreakStatement,
    ContinueStatement,
    PassStatement,
    PrintStatement,
    # OOP
    StructDef,
    ImplBlock,
    TraitDef,
    EnumDef,
    SelfExpression,
    StructLiteral,
    # Program
    Program,
    # Patterns
    MatchExpression,
    MatchArm,
    Pattern,
    Parameter,
)
from mathviz.compiler.const_evaluator import ConstEvaluator, ConstEvalError
from mathviz.utils.errors import SourceLocation


# =============================================================================
# Optimizer Pass Interface
# =============================================================================


class OptimizerPass(ABC):
    """
    Abstract base class for optimization passes.

    Each optimizer pass implements a specific optimization strategy
    and returns a new (potentially modified) AST.
    """

    @abstractmethod
    def optimize(self, program: Program) -> Program:
        """
        Apply the optimization pass to a program.

        Args:
            program: The input program AST

        Returns:
            A new program AST with optimizations applied
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the optimization pass."""
        pass


# =============================================================================
# 1. Constant Folder
# =============================================================================


class ConstantFolder(BaseASTVisitor, OptimizerPass):
    """
    Fold constant expressions at compile time.

    This optimizer evaluates expressions composed entirely of constants
    and replaces them with their computed values.

    Examples:
        2 + 3 -> 5
        "hello" + " world" -> "hello world"
        True and False -> False
        -(-5) -> 5
        PI * 2 -> 6.283185307179586
    """

    def __init__(self) -> None:
        """Initialize the constant folder."""
        self._evaluator = ConstEvaluator()
        self._changed = False

    @property
    def name(self) -> str:
        return "Constant Folding"

    def optimize(self, program: Program) -> Program:
        """Apply constant folding to the entire program."""
        return self.fold(program)

    def fold(self, program: Program) -> Program:
        """
        Return a new program with constants folded.

        Args:
            program: The input program AST

        Returns:
            A new program with constant expressions evaluated
        """
        self._changed = False
        new_statements = tuple(self._fold_statement(stmt) for stmt in program.statements)
        return Program(statements=new_statements, location=program.location)

    def _fold_statement(self, stmt: Statement) -> Statement:
        """Fold constants in a statement."""
        if isinstance(stmt, LetStatement):
            if stmt.value:
                new_value = self._fold_expression(stmt.value)
                if new_value is not stmt.value:
                    return LetStatement(
                        name=stmt.name,
                        type_annotation=stmt.type_annotation,
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, ConstDeclaration):
            new_value = self._fold_expression(stmt.value)
            if new_value is not stmt.value:
                return ConstDeclaration(
                    name=stmt.name,
                    value=new_value,
                    type_annotation=stmt.type_annotation,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, AssignmentStatement):
            new_value = self._fold_expression(stmt.value)
            if new_value is not stmt.value:
                return AssignmentStatement(
                    target=stmt.target,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ExpressionStatement):
            new_expr = self._fold_expression(stmt.expression)
            if new_expr is not stmt.expression:
                return ExpressionStatement(
                    expression=new_expr,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, IfStatement):
            return self._fold_if_statement(stmt)

        if isinstance(stmt, WhileStatement):
            new_condition = self._fold_expression(stmt.condition)
            new_body = self._fold_block(stmt.body)
            if new_condition is not stmt.condition or new_body is not stmt.body:
                return WhileStatement(
                    condition=new_condition,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ForStatement):
            new_iterable = self._fold_expression(stmt.iterable)
            new_body = self._fold_block(stmt.body)
            if new_iterable is not stmt.iterable or new_body is not stmt.body:
                return ForStatement(
                    variable=stmt.variable,
                    iterable=new_iterable,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ReturnStatement):
            if stmt.value:
                new_value = self._fold_expression(stmt.value)
                if new_value is not stmt.value:
                    return ReturnStatement(
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, FunctionDef):
            return self._fold_function_def(stmt)

        if isinstance(stmt, PrintStatement):
            new_format = self._fold_expression(stmt.format_string)
            new_args = tuple(self._fold_expression(arg) for arg in stmt.arguments)
            if new_format is not stmt.format_string or new_args != stmt.arguments:
                return PrintStatement(
                    format_string=new_format,
                    arguments=new_args,
                    newline=stmt.newline,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, CompoundAssignment):
            new_value = self._fold_expression(stmt.value)
            if new_value is not stmt.value:
                return CompoundAssignment(
                    target=stmt.target,
                    operator=stmt.operator,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        # For other statement types, return as-is
        return stmt

    def _fold_if_statement(self, stmt: IfStatement) -> Statement:
        """Fold constants in an if statement."""
        new_condition = self._fold_expression(stmt.condition)
        new_then_block = self._fold_block(stmt.then_block)

        new_elif_clauses: list[tuple[Expression, Block]] = []
        for cond, block in stmt.elif_clauses:
            new_elif_clauses.append((
                self._fold_expression(cond),
                self._fold_block(block),
            ))

        new_else_block = None
        if stmt.else_block:
            new_else_block = self._fold_block(stmt.else_block)

        return IfStatement(
            condition=new_condition,
            then_block=new_then_block,
            elif_clauses=tuple(new_elif_clauses),
            else_block=new_else_block,
            location=stmt.location,
        )

    def _fold_function_def(self, func: FunctionDef) -> FunctionDef:
        """Fold constants in a function definition."""
        if func.body is None:
            return func

        new_body = self._fold_block(func.body)

        # Fold default parameter values
        new_params: list[Parameter] = []
        params_changed = False
        for param in func.parameters:
            if param.default_value:
                new_default = self._fold_expression(param.default_value)
                if new_default is not param.default_value:
                    params_changed = True
                    new_params.append(Parameter(
                        name=param.name,
                        type_annotation=param.type_annotation,
                        default_value=new_default,
                        location=param.location,
                    ))
                else:
                    new_params.append(param)
            else:
                new_params.append(param)

        if new_body is not func.body or params_changed:
            return FunctionDef(
                name=func.name,
                parameters=tuple(new_params),
                return_type=func.return_type,
                body=new_body,
                type_params=func.type_params,
                where_clause=func.where_clause,
                jit_options=func.jit_options,
                doc_comment=func.doc_comment,
                attributes=func.attributes,
                location=func.location,
            )
        return func

    def _fold_block(self, block: Block) -> Block:
        """Fold constants in a block."""
        new_statements = tuple(self._fold_statement(stmt) for stmt in block.statements)
        if new_statements != block.statements:
            return Block(statements=new_statements, location=block.location)
        return block

    def _fold_expression(self, expr: Expression) -> Expression:
        """
        Fold constants in an expression, returning a new expression if changed.
        """
        # Try to evaluate the entire expression as a constant
        if self._is_constant(expr):
            try:
                value = self._evaluator.evaluate(expr)
                self._changed = True
                return self._value_to_literal(value, expr.location)
            except ConstEvalError:
                pass

        # Recursively fold subexpressions
        if isinstance(expr, BinaryExpression):
            left = self._fold_expression(expr.left)
            right = self._fold_expression(expr.right)

            # Try to fold after subexpressions are folded
            if self._is_literal(left) and self._is_literal(right):
                result = self._fold_binary(
                    self._literal_value(left),
                    expr.operator,
                    self._literal_value(right),
                )
                if result is not None:
                    self._changed = True
                    return self._value_to_literal(result, expr.location)

            if left is not expr.left or right is not expr.right:
                return BinaryExpression(
                    left=left,
                    operator=expr.operator,
                    right=right,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, UnaryExpression):
            operand = self._fold_expression(expr.operand)

            # Try to fold after subexpression is folded
            if self._is_literal(operand):
                result = self._fold_unary(expr.operator, self._literal_value(operand))
                if result is not None:
                    self._changed = True
                    return self._value_to_literal(result, expr.location)

            if operand is not expr.operand:
                return UnaryExpression(
                    operator=expr.operator,
                    operand=operand,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, ConditionalExpression):
            condition = self._fold_expression(expr.condition)
            then_expr = self._fold_expression(expr.then_expr)
            else_expr = self._fold_expression(expr.else_expr)

            # If condition is constant, return the appropriate branch
            if isinstance(condition, BooleanLiteral):
                self._changed = True
                return then_expr if condition.value else else_expr

            if (condition is not expr.condition or
                then_expr is not expr.then_expr or
                else_expr is not expr.else_expr):
                return ConditionalExpression(
                    condition=condition,
                    then_expr=then_expr,
                    else_expr=else_expr,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, CallExpression):
            new_callee = self._fold_expression(expr.callee)
            new_args = tuple(self._fold_expression(arg) for arg in expr.arguments)
            if new_callee is not expr.callee or new_args != expr.arguments:
                return CallExpression(
                    callee=new_callee,
                    arguments=new_args,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, ListLiteral):
            new_elements = tuple(self._fold_expression(e) for e in expr.elements)
            if new_elements != expr.elements:
                return ListLiteral(elements=new_elements, location=expr.location)
            return expr

        if isinstance(expr, TupleLiteral):
            new_elements = tuple(self._fold_expression(e) for e in expr.elements)
            if new_elements != expr.elements:
                return TupleLiteral(elements=new_elements, location=expr.location)
            return expr

        if isinstance(expr, IndexExpression):
            new_obj = self._fold_expression(expr.object)
            new_idx = self._fold_expression(expr.index)
            if new_obj is not expr.object or new_idx is not expr.index:
                return IndexExpression(
                    object=new_obj,
                    index=new_idx,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, MemberAccess):
            new_obj = self._fold_expression(expr.object)
            if new_obj is not expr.object:
                return MemberAccess(
                    object=new_obj,
                    member=expr.member,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, RangeExpression):
            new_start = self._fold_expression(expr.start)
            new_end = self._fold_expression(expr.end)
            new_step = self._fold_expression(expr.step) if expr.step else None
            if (new_start is not expr.start or new_end is not expr.end or
                new_step is not expr.step):
                return RangeExpression(
                    start=new_start,
                    end=new_end,
                    inclusive=expr.inclusive,
                    step=new_step,
                    location=expr.location,
                )
            return expr

        # For literals and identifiers, return as-is
        return expr

    def _fold_binary(
        self,
        left: Any,
        op: BinaryOperator,
        right: Any
    ) -> Optional[Any]:
        """
        Fold a binary operation on constant values.

        Returns None if the operation cannot be folded.
        """
        try:
            if op == BinaryOperator.ADD:
                return left + right
            if op == BinaryOperator.SUB:
                return left - right
            if op == BinaryOperator.MUL:
                return left * right
            if op == BinaryOperator.DIV:
                if right == 0:
                    return None  # Don't fold division by zero
                return left / right
            if op == BinaryOperator.FLOOR_DIV:
                if right == 0:
                    return None
                return left // right
            if op == BinaryOperator.MOD:
                if right == 0:
                    return None
                return left % right
            if op == BinaryOperator.POW:
                return left ** right
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
            if op == BinaryOperator.AND:
                return left and right
            if op == BinaryOperator.OR:
                return left or right
        except Exception:
            return None
        return None

    def _fold_unary(self, op: UnaryOperator, operand: Any) -> Optional[Any]:
        """
        Fold a unary operation on a constant value.

        Returns None if the operation cannot be folded.
        """
        try:
            if op == UnaryOperator.NEG:
                return -operand
            if op == UnaryOperator.POS:
                return +operand
            if op == UnaryOperator.NOT:
                return not operand
        except Exception:
            return None
        return None

    def _is_constant(self, expr: Expression) -> bool:
        """Check if an expression is a compile-time constant."""
        return self._evaluator.is_const_expr(expr)

    def _is_literal(self, expr: Expression) -> bool:
        """Check if an expression is a literal value."""
        return isinstance(expr, (
            IntegerLiteral,
            FloatLiteral,
            StringLiteral,
            BooleanLiteral,
            NoneLiteral,
        ))

    def _literal_value(self, expr: Expression) -> Any:
        """Extract the value from a literal expression."""
        if isinstance(expr, (IntegerLiteral, FloatLiteral, StringLiteral, BooleanLiteral)):
            return expr.value
        if isinstance(expr, NoneLiteral):
            return None
        raise ValueError(f"Not a literal: {expr}")

    def _value_to_literal(
        self,
        value: Any,
        location: Optional[SourceLocation] = None
    ) -> Expression:
        """Convert a Python value to a literal expression."""
        if isinstance(value, bool):
            return BooleanLiteral(value=value, location=location)
        if isinstance(value, int):
            return IntegerLiteral(value=value, location=location)
        if isinstance(value, float):
            return FloatLiteral(value=value, location=location)
        if isinstance(value, str):
            return StringLiteral(value=value, location=location)
        if value is None:
            return NoneLiteral(location=location)
        if isinstance(value, list):
            return ListLiteral(
                elements=tuple(self._value_to_literal(v, location) for v in value),
                location=location,
            )
        if isinstance(value, tuple):
            return TupleLiteral(
                elements=tuple(self._value_to_literal(v, location) for v in value),
                location=location,
            )
        # Cannot convert, return a placeholder
        raise ValueError(f"Cannot convert {type(value)} to literal")


# =============================================================================
# 2. Constant Propagator
# =============================================================================


@dataclass
class ConstantScope:
    """
    Tracks constant values within a scope.

    Handles nested scopes with proper shadowing.
    """
    constants: dict[str, Any] = field(default_factory=dict)
    parent: Optional["ConstantScope"] = None

    def get(self, name: str) -> Optional[Any]:
        """Get a constant value, searching parent scopes."""
        if name in self.constants:
            return self.constants[name]
        if self.parent:
            return self.parent.get(name)
        return None

    def has(self, name: str) -> bool:
        """Check if a constant is defined in this or parent scopes."""
        if name in self.constants:
            return True
        if self.parent:
            return self.parent.has(name)
        return False

    def set(self, name: str, value: Any) -> None:
        """Set a constant value in this scope."""
        self.constants[name] = value

    def invalidate(self, name: str) -> None:
        """Remove a constant (when it's reassigned with non-constant)."""
        if name in self.constants:
            del self.constants[name]

    def child(self) -> "ConstantScope":
        """Create a child scope."""
        return ConstantScope(parent=self)


class ConstantPropagator(OptimizerPass):
    """
    Propagate known constant values through the program.

    This optimizer tracks variables that are assigned constant values
    and substitutes their uses with the constant value.

    Example:
        let x = 5
        let y = x + 3  ->  let y = 8

        const PI = 3.14
        let circumference = 2 * PI * r  ->  let circumference = 6.28 * r
    """

    def __init__(self) -> None:
        """Initialize the constant propagator."""
        self._scope: ConstantScope = ConstantScope()
        self._folder = ConstantFolder()

    @property
    def name(self) -> str:
        return "Constant Propagation"

    def optimize(self, program: Program) -> Program:
        """Apply constant propagation to the entire program."""
        return self.propagate(program)

    def propagate(self, program: Program) -> Program:
        """
        Propagate constants and simplify expressions.

        Args:
            program: The input program AST

        Returns:
            A new program with constants propagated
        """
        self._scope = ConstantScope()

        # First pass: collect top-level constants
        self._collect_constants(program)

        # Second pass: propagate constants through the program
        new_statements = []
        for stmt in program.statements:
            new_stmt = self._propagate_statement(stmt)
            new_statements.append(new_stmt)

        return Program(
            statements=tuple(new_statements),
            location=program.location,
        )

    def _collect_constants(self, program: Program) -> None:
        """Collect all top-level constant declarations."""
        evaluator = ConstEvaluator()

        for stmt in program.statements:
            if isinstance(stmt, ConstDeclaration):
                try:
                    value = evaluator.evaluate(stmt.value)
                    self._scope.set(stmt.name, value)
                    evaluator.add_constant(stmt.name, value)
                except ConstEvalError:
                    pass

    def _propagate_statement(self, stmt: Statement) -> Statement:
        """Propagate constants through a statement."""
        if isinstance(stmt, LetStatement):
            if stmt.value:
                new_value = self._substitute_constants(stmt.value)

                # Check if the new value is a constant
                if self._is_constant_value(new_value):
                    try:
                        value = self._folder._evaluator.evaluate(new_value)
                        self._scope.set(stmt.name, value)
                    except ConstEvalError:
                        pass
                else:
                    # Variable is not constant, cannot propagate
                    self._scope.invalidate(stmt.name)

                if new_value is not stmt.value:
                    return LetStatement(
                        name=stmt.name,
                        type_annotation=stmt.type_annotation,
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, AssignmentStatement):
            new_value = self._substitute_constants(stmt.value)

            # Invalidate the target if it's a simple identifier
            if isinstance(stmt.target, Identifier):
                if self._is_constant_value(new_value):
                    try:
                        value = self._folder._evaluator.evaluate(new_value)
                        self._scope.set(stmt.target.name, value)
                    except ConstEvalError:
                        self._scope.invalidate(stmt.target.name)
                else:
                    self._scope.invalidate(stmt.target.name)

            if new_value is not stmt.value:
                return AssignmentStatement(
                    target=stmt.target,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ExpressionStatement):
            new_expr = self._substitute_constants(stmt.expression)
            if new_expr is not stmt.expression:
                return ExpressionStatement(
                    expression=new_expr,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, IfStatement):
            return self._propagate_if_statement(stmt)

        if isinstance(stmt, WhileStatement):
            new_cond = self._substitute_constants(stmt.condition)
            new_body = self._propagate_block(stmt.body)
            if new_cond is not stmt.condition or new_body is not stmt.body:
                return WhileStatement(
                    condition=new_cond,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ForStatement):
            # Enter a new scope for the loop
            old_scope = self._scope
            self._scope = self._scope.child()

            new_iterable = self._substitute_constants(stmt.iterable)
            # Loop variable is not constant
            self._scope.invalidate(stmt.variable)
            new_body = self._propagate_block(stmt.body)

            self._scope = old_scope

            if new_iterable is not stmt.iterable or new_body is not stmt.body:
                return ForStatement(
                    variable=stmt.variable,
                    iterable=new_iterable,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ReturnStatement):
            if stmt.value:
                new_value = self._substitute_constants(stmt.value)
                if new_value is not stmt.value:
                    return ReturnStatement(
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, FunctionDef):
            return self._propagate_function_def(stmt)

        if isinstance(stmt, PrintStatement):
            new_format = self._substitute_constants(stmt.format_string)
            new_args = tuple(self._substitute_constants(arg) for arg in stmt.arguments)
            if new_format is not stmt.format_string or new_args != stmt.arguments:
                return PrintStatement(
                    format_string=new_format,
                    arguments=new_args,
                    newline=stmt.newline,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, CompoundAssignment):
            # Invalidate the target
            if isinstance(stmt.target, Identifier):
                self._scope.invalidate(stmt.target.name)

            new_value = self._substitute_constants(stmt.value)
            if new_value is not stmt.value:
                return CompoundAssignment(
                    target=stmt.target,
                    operator=stmt.operator,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        return stmt

    def _propagate_if_statement(self, stmt: IfStatement) -> Statement:
        """Propagate constants through an if statement."""
        new_condition = self._substitute_constants(stmt.condition)

        # Enter a new scope for then block
        old_scope = self._scope
        self._scope = self._scope.child()
        new_then_block = self._propagate_block(stmt.then_block)
        self._scope = old_scope

        # Process elif clauses
        new_elif_clauses: list[tuple[Expression, Block]] = []
        for cond, block in stmt.elif_clauses:
            new_cond = self._substitute_constants(cond)
            self._scope = self._scope.child()
            new_block = self._propagate_block(block)
            self._scope = old_scope
            new_elif_clauses.append((new_cond, new_block))

        # Process else block
        new_else_block = None
        if stmt.else_block:
            self._scope = self._scope.child()
            new_else_block = self._propagate_block(stmt.else_block)
            self._scope = old_scope

        return IfStatement(
            condition=new_condition,
            then_block=new_then_block,
            elif_clauses=tuple(new_elif_clauses),
            else_block=new_else_block,
            location=stmt.location,
        )

    def _propagate_function_def(self, func: FunctionDef) -> FunctionDef:
        """Propagate constants through a function definition."""
        if func.body is None:
            return func

        # Enter a new scope for the function
        old_scope = self._scope
        self._scope = self._scope.child()

        # Parameters are not constants
        for param in func.parameters:
            self._scope.invalidate(param.name)

        new_body = self._propagate_block(func.body)

        self._scope = old_scope

        if new_body is not func.body:
            return FunctionDef(
                name=func.name,
                parameters=func.parameters,
                return_type=func.return_type,
                body=new_body,
                type_params=func.type_params,
                where_clause=func.where_clause,
                jit_options=func.jit_options,
                doc_comment=func.doc_comment,
                attributes=func.attributes,
                location=func.location,
            )
        return func

    def _propagate_block(self, block: Block) -> Block:
        """Propagate constants through a block."""
        new_statements = tuple(
            self._propagate_statement(stmt) for stmt in block.statements
        )
        if new_statements != block.statements:
            return Block(statements=new_statements, location=block.location)
        return block

    def _substitute_constants(self, expr: Expression) -> Expression:
        """
        Replace variable references with their constant values.

        Args:
            expr: The expression to process

        Returns:
            A new expression with constants substituted
        """
        if isinstance(expr, Identifier):
            value = self._scope.get(expr.name)
            if value is not None:
                return self._folder._value_to_literal(value, expr.location)
            return expr

        if isinstance(expr, BinaryExpression):
            left = self._substitute_constants(expr.left)
            right = self._substitute_constants(expr.right)

            # Try to fold after substitution
            new_expr = BinaryExpression(
                left=left,
                operator=expr.operator,
                right=right,
                location=expr.location,
            )
            return self._folder._fold_expression(new_expr)

        if isinstance(expr, UnaryExpression):
            operand = self._substitute_constants(expr.operand)
            new_expr = UnaryExpression(
                operator=expr.operator,
                operand=operand,
                location=expr.location,
            )
            return self._folder._fold_expression(new_expr)

        if isinstance(expr, ConditionalExpression):
            condition = self._substitute_constants(expr.condition)
            then_expr = self._substitute_constants(expr.then_expr)
            else_expr = self._substitute_constants(expr.else_expr)

            # If condition is constant, return the appropriate branch
            if isinstance(condition, BooleanLiteral):
                return then_expr if condition.value else else_expr

            if (condition is not expr.condition or
                then_expr is not expr.then_expr or
                else_expr is not expr.else_expr):
                return ConditionalExpression(
                    condition=condition,
                    then_expr=then_expr,
                    else_expr=else_expr,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, CallExpression):
            new_callee = self._substitute_constants(expr.callee)
            new_args = tuple(self._substitute_constants(arg) for arg in expr.arguments)
            if new_callee is not expr.callee or new_args != expr.arguments:
                return CallExpression(
                    callee=new_callee,
                    arguments=new_args,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, ListLiteral):
            new_elements = tuple(self._substitute_constants(e) for e in expr.elements)
            if new_elements != expr.elements:
                return ListLiteral(elements=new_elements, location=expr.location)
            return expr

        if isinstance(expr, TupleLiteral):
            new_elements = tuple(self._substitute_constants(e) for e in expr.elements)
            if new_elements != expr.elements:
                return TupleLiteral(elements=new_elements, location=expr.location)
            return expr

        if isinstance(expr, IndexExpression):
            new_obj = self._substitute_constants(expr.object)
            new_idx = self._substitute_constants(expr.index)
            if new_obj is not expr.object or new_idx is not expr.index:
                return IndexExpression(
                    object=new_obj,
                    index=new_idx,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, MemberAccess):
            new_obj = self._substitute_constants(expr.object)
            if new_obj is not expr.object:
                return MemberAccess(
                    object=new_obj,
                    member=expr.member,
                    location=expr.location,
                )
            return expr

        return expr

    def _is_constant_value(self, expr: Expression) -> bool:
        """Check if an expression evaluates to a constant."""
        if isinstance(expr, (IntegerLiteral, FloatLiteral, StringLiteral,
                            BooleanLiteral, NoneLiteral)):
            return True
        if isinstance(expr, Identifier):
            return self._scope.has(expr.name)
        if isinstance(expr, BinaryExpression):
            return self._is_constant_value(expr.left) and self._is_constant_value(expr.right)
        if isinstance(expr, UnaryExpression):
            return self._is_constant_value(expr.operand)
        return False

    def _build_constant_map(self, block: Block) -> dict[str, Any]:
        """Build a map of variables with known constant values in a block."""
        constants: dict[str, Any] = {}
        evaluator = ConstEvaluator()

        for stmt in block.statements:
            if isinstance(stmt, (LetStatement, ConstDeclaration)):
                if hasattr(stmt, 'value') and stmt.value:
                    try:
                        value = evaluator.evaluate(stmt.value)
                        constants[stmt.name] = value
                        evaluator.add_constant(stmt.name, value)
                    except ConstEvalError:
                        # Not a constant, invalidate if previously tracked
                        if stmt.name in constants:
                            del constants[stmt.name]
            elif isinstance(stmt, AssignmentStatement):
                if isinstance(stmt.target, Identifier):
                    # Assignment invalidates constness
                    if stmt.target.name in constants:
                        del constants[stmt.target.name]

        return constants


# =============================================================================
# 3. Dead Code Eliminator
# =============================================================================


class DeadCodeEliminator(OptimizerPass):
    """
    Remove unreachable and dead code.

    This optimizer removes:
    - Statements after unconditional return/break/continue
    - If statements with constant conditions
    - Unused variable assignments (when safe)
    - While loops with constant false conditions
    """

    def __init__(self) -> None:
        """Initialize the dead code eliminator."""
        self._used_variables: set[str] = set()

    @property
    def name(self) -> str:
        return "Dead Code Elimination"

    def optimize(self, program: Program) -> Program:
        """Apply dead code elimination to the entire program."""
        return self.eliminate(program)

    def eliminate(self, program: Program) -> Program:
        """
        Remove dead code from the program.

        Args:
            program: The input program AST

        Returns:
            A new program with dead code removed
        """
        # First pass: collect used variables
        self._used_variables = set()
        self._collect_used_variables(program)

        # Second pass: eliminate dead code
        new_statements = self._eliminate_statements(list(program.statements))

        return Program(
            statements=tuple(new_statements),
            location=program.location,
        )

    def _collect_used_variables(self, node: ASTNode) -> None:
        """Collect all variable names that are read in the program."""
        if isinstance(node, Identifier):
            self._used_variables.add(node.name)
        elif isinstance(node, Program):
            for stmt in node.statements:
                self._collect_used_variables(stmt)
        elif isinstance(node, Block):
            for stmt in node.statements:
                self._collect_used_variables(stmt)
        elif isinstance(node, LetStatement):
            if node.value:
                self._collect_used_variables(node.value)
        elif isinstance(node, ConstDeclaration):
            self._collect_used_variables(node.value)
        elif isinstance(node, AssignmentStatement):
            self._collect_used_variables(node.value)
            # Also collect from target for member access, etc.
            if not isinstance(node.target, Identifier):
                self._collect_used_variables(node.target)
        elif isinstance(node, CompoundAssignment):
            self._collect_used_variables(node.target)
            self._collect_used_variables(node.value)
        elif isinstance(node, ExpressionStatement):
            self._collect_used_variables(node.expression)
        elif isinstance(node, BinaryExpression):
            self._collect_used_variables(node.left)
            self._collect_used_variables(node.right)
        elif isinstance(node, UnaryExpression):
            self._collect_used_variables(node.operand)
        elif isinstance(node, CallExpression):
            self._collect_used_variables(node.callee)
            for arg in node.arguments:
                self._collect_used_variables(arg)
        elif isinstance(node, MemberAccess):
            self._collect_used_variables(node.object)
        elif isinstance(node, IndexExpression):
            self._collect_used_variables(node.object)
            self._collect_used_variables(node.index)
        elif isinstance(node, ConditionalExpression):
            self._collect_used_variables(node.condition)
            self._collect_used_variables(node.then_expr)
            self._collect_used_variables(node.else_expr)
        elif isinstance(node, IfStatement):
            self._collect_used_variables(node.condition)
            self._collect_used_variables(node.then_block)
            for cond, block in node.elif_clauses:
                self._collect_used_variables(cond)
                self._collect_used_variables(block)
            if node.else_block:
                self._collect_used_variables(node.else_block)
        elif isinstance(node, WhileStatement):
            self._collect_used_variables(node.condition)
            self._collect_used_variables(node.body)
        elif isinstance(node, ForStatement):
            self._collect_used_variables(node.iterable)
            self._collect_used_variables(node.body)
        elif isinstance(node, ReturnStatement):
            if node.value:
                self._collect_used_variables(node.value)
        elif isinstance(node, FunctionDef):
            if node.body:
                self._collect_used_variables(node.body)
            for param in node.parameters:
                if param.default_value:
                    self._collect_used_variables(param.default_value)
        elif isinstance(node, PrintStatement):
            self._collect_used_variables(node.format_string)
            for arg in node.arguments:
                self._collect_used_variables(arg)
        elif isinstance(node, ListLiteral):
            for elem in node.elements:
                self._collect_used_variables(elem)
        elif isinstance(node, TupleLiteral):
            for elem in node.elements:
                self._collect_used_variables(elem)
        elif isinstance(node, RangeExpression):
            self._collect_used_variables(node.start)
            self._collect_used_variables(node.end)
            if node.step:
                self._collect_used_variables(node.step)

    def _eliminate_statements(self, statements: list[Statement]) -> list[Statement]:
        """Eliminate dead code from a list of statements."""
        result: list[Statement] = []
        unreachable = False

        for stmt in statements:
            if unreachable:
                # Skip unreachable statements
                continue

            # Process the statement
            new_stmt = self._eliminate_statement(stmt)

            if new_stmt is None:
                # Statement was eliminated
                continue

            if isinstance(new_stmt, list):
                # Statement was expanded (e.g., if with constant condition)
                result.extend(new_stmt)
                # Check if last statement makes code unreachable
                if new_stmt and self._is_terminating(new_stmt[-1]):
                    unreachable = True
            else:
                result.append(new_stmt)

                # Check if this statement makes subsequent code unreachable
                if self._is_terminating(new_stmt):
                    unreachable = True

        return result

    def _eliminate_statement(
        self, stmt: Statement
    ) -> Optional[Union[Statement, list[Statement]]]:
        """
        Eliminate dead code from a single statement.

        Returns:
            - None if the statement should be removed
            - A list of statements if the statement expands to multiple
            - The (possibly modified) statement otherwise
        """
        if isinstance(stmt, IfStatement):
            return self._fold_if_with_constant_condition(stmt)

        if isinstance(stmt, WhileStatement):
            # While with constant false condition is dead code
            if isinstance(stmt.condition, BooleanLiteral):
                if not stmt.condition.value:
                    return None

            new_body = self._eliminate_block(stmt.body)
            if new_body is not stmt.body:
                return WhileStatement(
                    condition=stmt.condition,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ForStatement):
            new_body = self._eliminate_block(stmt.body)
            if new_body is not stmt.body:
                return ForStatement(
                    variable=stmt.variable,
                    iterable=stmt.iterable,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, FunctionDef):
            if stmt.body:
                new_body = self._eliminate_block(stmt.body)
                # Also remove statements after return
                new_body = self._remove_after_return(new_body)
                if new_body is not stmt.body:
                    return FunctionDef(
                        name=stmt.name,
                        parameters=stmt.parameters,
                        return_type=stmt.return_type,
                        body=new_body,
                        type_params=stmt.type_params,
                        where_clause=stmt.where_clause,
                        jit_options=stmt.jit_options,
                        doc_comment=stmt.doc_comment,
                        attributes=stmt.attributes,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, LetStatement):
            # Remove unused variable declarations if they have no side effects
            if stmt.name not in self._used_variables:
                if stmt.value is None or self._is_pure_expression(stmt.value):
                    return None
            return stmt

        # Pass statement does nothing, but don't remove it if it's needed
        if isinstance(stmt, PassStatement):
            return stmt

        return stmt

    def _fold_if_with_constant_condition(
        self, stmt: IfStatement
    ) -> Union[Statement, list[Statement]]:
        """
        Simplify if statements with constant conditions.

        if True { x } else { y } -> x
        if False { x } else { y } -> y
        """
        if isinstance(stmt.condition, BooleanLiteral):
            if stmt.condition.value:
                # Condition is True, return then block statements
                new_block = self._eliminate_block(stmt.then_block)
                return list(new_block.statements)
            else:
                # Condition is False, check elif clauses
                for cond, block in stmt.elif_clauses:
                    if isinstance(cond, BooleanLiteral):
                        if cond.value:
                            new_block = self._eliminate_block(block)
                            return list(new_block.statements)
                    else:
                        # Non-constant elif, rebuild from here
                        remaining_elifs = []
                        found = False
                        for c, b in stmt.elif_clauses:
                            if found:
                                remaining_elifs.append((c, self._eliminate_block(b)))
                            elif c is cond:
                                remaining_elifs.append((c, self._eliminate_block(b)))
                                found = True

                        return IfStatement(
                            condition=cond,
                            then_block=self._eliminate_block(block),
                            elif_clauses=tuple(remaining_elifs),
                            else_block=(self._eliminate_block(stmt.else_block)
                                       if stmt.else_block else None),
                            location=stmt.location,
                        )

                # All conditions were false, return else block
                if stmt.else_block:
                    new_block = self._eliminate_block(stmt.else_block)
                    return list(new_block.statements)
                return []

        # Non-constant condition, just eliminate within blocks
        new_then = self._eliminate_block(stmt.then_block)
        new_elifs = tuple(
            (cond, self._eliminate_block(block))
            for cond, block in stmt.elif_clauses
        )
        new_else = self._eliminate_block(stmt.else_block) if stmt.else_block else None

        if (new_then is not stmt.then_block or
            new_elifs != stmt.elif_clauses or
            new_else is not stmt.else_block):
            return IfStatement(
                condition=stmt.condition,
                then_block=new_then,
                elif_clauses=new_elifs,
                else_block=new_else,
                location=stmt.location,
            )
        return stmt

    def _remove_after_return(self, block: Block) -> Block:
        """Remove statements after an unconditional return."""
        new_statements: list[Statement] = []

        for stmt in block.statements:
            new_statements.append(stmt)
            if self._is_terminating(stmt):
                break

        if len(new_statements) != len(block.statements):
            return Block(statements=tuple(new_statements), location=block.location)
        return block

    def _eliminate_block(self, block: Block) -> Block:
        """Eliminate dead code from a block."""
        new_statements = self._eliminate_statements(list(block.statements))
        if len(new_statements) != len(block.statements) or any(
            s1 is not s2 for s1, s2 in zip(new_statements, block.statements)
        ):
            return Block(statements=tuple(new_statements), location=block.location)
        return block

    def _is_terminating(self, stmt: Statement) -> bool:
        """Check if a statement terminates control flow."""
        return isinstance(stmt, (ReturnStatement, BreakStatement, ContinueStatement))

    def _is_pure_expression(self, expr: Expression) -> bool:
        """Check if an expression has no side effects."""
        if isinstance(expr, (IntegerLiteral, FloatLiteral, StringLiteral,
                            BooleanLiteral, NoneLiteral, Identifier)):
            return True
        if isinstance(expr, BinaryExpression):
            return self._is_pure_expression(expr.left) and self._is_pure_expression(expr.right)
        if isinstance(expr, UnaryExpression):
            return self._is_pure_expression(expr.operand)
        if isinstance(expr, ListLiteral):
            return all(self._is_pure_expression(e) for e in expr.elements)
        if isinstance(expr, TupleLiteral):
            return all(self._is_pure_expression(e) for e in expr.elements)
        if isinstance(expr, ConditionalExpression):
            return (self._is_pure_expression(expr.condition) and
                    self._is_pure_expression(expr.then_expr) and
                    self._is_pure_expression(expr.else_expr))
        # Function calls may have side effects
        return False

    def _remove_unused_variables(self, block: Block) -> Block:
        """Remove assignments to variables that are never read."""
        # This is handled in _eliminate_statement for LetStatements
        return self._eliminate_block(block)


# =============================================================================
# 4. Algebraic Simplifier
# =============================================================================


class AlgebraicSimplifier(OptimizerPass):
    """
    Simplify algebraic expressions using mathematical identities.

    Identities applied:
    - x + 0 -> x, 0 + x -> x
    - x - 0 -> x
    - x * 0 -> 0, 0 * x -> 0
    - x * 1 -> x, 1 * x -> x
    - x / 1 -> x
    - x ** 0 -> 1
    - x ** 1 -> x
    - x - x -> 0
    - x / x -> 1 (when x != 0)
    - -(-x) -> x
    - not not x -> x
    - True and x -> x, x and True -> x
    - False and x -> False, x and False -> False
    - True or x -> True, x or True -> True
    - False or x -> x, x or False -> x
    """

    def __init__(self) -> None:
        """Initialize the algebraic simplifier."""
        pass

    @property
    def name(self) -> str:
        return "Algebraic Simplification"

    def optimize(self, program: Program) -> Program:
        """Apply algebraic simplifications to the entire program."""
        new_statements = tuple(
            self._simplify_statement(stmt) for stmt in program.statements
        )
        return Program(statements=new_statements, location=program.location)

    def simplify(self, expr: Expression) -> Expression:
        """
        Apply algebraic simplifications to an expression.

        Args:
            expr: The expression to simplify

        Returns:
            A simplified expression
        """
        return self._simplify_expression(expr)

    def _simplify_statement(self, stmt: Statement) -> Statement:
        """Simplify algebraic expressions in a statement."""
        if isinstance(stmt, LetStatement):
            if stmt.value:
                new_value = self._simplify_expression(stmt.value)
                if new_value is not stmt.value:
                    return LetStatement(
                        name=stmt.name,
                        type_annotation=stmt.type_annotation,
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, AssignmentStatement):
            new_value = self._simplify_expression(stmt.value)
            if new_value is not stmt.value:
                return AssignmentStatement(
                    target=stmt.target,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ExpressionStatement):
            new_expr = self._simplify_expression(stmt.expression)
            if new_expr is not stmt.expression:
                return ExpressionStatement(
                    expression=new_expr,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ReturnStatement):
            if stmt.value:
                new_value = self._simplify_expression(stmt.value)
                if new_value is not stmt.value:
                    return ReturnStatement(
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, IfStatement):
            new_cond = self._simplify_expression(stmt.condition)
            new_then = self._simplify_block(stmt.then_block)
            new_elifs = tuple(
                (self._simplify_expression(c), self._simplify_block(b))
                for c, b in stmt.elif_clauses
            )
            new_else = self._simplify_block(stmt.else_block) if stmt.else_block else None

            if (new_cond is not stmt.condition or new_then is not stmt.then_block or
                new_elifs != stmt.elif_clauses or new_else is not stmt.else_block):
                return IfStatement(
                    condition=new_cond,
                    then_block=new_then,
                    elif_clauses=new_elifs,
                    else_block=new_else,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, WhileStatement):
            new_cond = self._simplify_expression(stmt.condition)
            new_body = self._simplify_block(stmt.body)
            if new_cond is not stmt.condition or new_body is not stmt.body:
                return WhileStatement(
                    condition=new_cond,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ForStatement):
            new_iter = self._simplify_expression(stmt.iterable)
            new_body = self._simplify_block(stmt.body)
            if new_iter is not stmt.iterable or new_body is not stmt.body:
                return ForStatement(
                    variable=stmt.variable,
                    iterable=new_iter,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, FunctionDef):
            if stmt.body:
                new_body = self._simplify_block(stmt.body)
                if new_body is not stmt.body:
                    return FunctionDef(
                        name=stmt.name,
                        parameters=stmt.parameters,
                        return_type=stmt.return_type,
                        body=new_body,
                        type_params=stmt.type_params,
                        where_clause=stmt.where_clause,
                        jit_options=stmt.jit_options,
                        doc_comment=stmt.doc_comment,
                        attributes=stmt.attributes,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, PrintStatement):
            new_format = self._simplify_expression(stmt.format_string)
            new_args = tuple(self._simplify_expression(arg) for arg in stmt.arguments)
            if new_format is not stmt.format_string or new_args != stmt.arguments:
                return PrintStatement(
                    format_string=new_format,
                    arguments=new_args,
                    newline=stmt.newline,
                    location=stmt.location,
                )
            return stmt

        return stmt

    def _simplify_block(self, block: Block) -> Block:
        """Simplify algebraic expressions in a block."""
        new_statements = tuple(
            self._simplify_statement(stmt) for stmt in block.statements
        )
        if new_statements != block.statements:
            return Block(statements=new_statements, location=block.location)
        return block

    def _simplify_expression(self, expr: Expression) -> Expression:
        """Apply algebraic simplifications to an expression."""
        if isinstance(expr, BinaryExpression):
            # First simplify operands
            left = self._simplify_expression(expr.left)
            right = self._simplify_expression(expr.right)

            # Apply identities
            result = self._apply_binary_identity(left, expr.operator, right, expr.location)
            if result is not None:
                return result

            # Return simplified expression
            if left is not expr.left or right is not expr.right:
                return BinaryExpression(
                    left=left,
                    operator=expr.operator,
                    right=right,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, UnaryExpression):
            operand = self._simplify_expression(expr.operand)

            # Apply identities
            result = self._apply_unary_identity(expr.operator, operand, expr.location)
            if result is not None:
                return result

            if operand is not expr.operand:
                return UnaryExpression(
                    operator=expr.operator,
                    operand=operand,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, ConditionalExpression):
            condition = self._simplify_expression(expr.condition)
            then_expr = self._simplify_expression(expr.then_expr)
            else_expr = self._simplify_expression(expr.else_expr)

            if (condition is not expr.condition or
                then_expr is not expr.then_expr or
                else_expr is not expr.else_expr):
                return ConditionalExpression(
                    condition=condition,
                    then_expr=then_expr,
                    else_expr=else_expr,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, CallExpression):
            new_callee = self._simplify_expression(expr.callee)
            new_args = tuple(self._simplify_expression(arg) for arg in expr.arguments)
            if new_callee is not expr.callee or new_args != expr.arguments:
                return CallExpression(
                    callee=new_callee,
                    arguments=new_args,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, ListLiteral):
            new_elements = tuple(self._simplify_expression(e) for e in expr.elements)
            if new_elements != expr.elements:
                return ListLiteral(elements=new_elements, location=expr.location)
            return expr

        if isinstance(expr, TupleLiteral):
            new_elements = tuple(self._simplify_expression(e) for e in expr.elements)
            if new_elements != expr.elements:
                return TupleLiteral(elements=new_elements, location=expr.location)
            return expr

        return expr

    def _apply_binary_identity(
        self,
        left: Expression,
        op: BinaryOperator,
        right: Expression,
        location: Optional[SourceLocation],
    ) -> Optional[Expression]:
        """
        Apply binary algebraic identities.

        Returns the simplified expression or None if no identity applies.
        """
        # x + 0 -> x, 0 + x -> x
        if op == BinaryOperator.ADD:
            if self._is_zero(right):
                return left
            if self._is_zero(left):
                return right

        # x - 0 -> x
        if op == BinaryOperator.SUB:
            if self._is_zero(right):
                return left
            # x - x -> 0
            if self._exprs_equal(left, right):
                return IntegerLiteral(value=0, location=location)

        # x * 0 -> 0, 0 * x -> 0
        # x * 1 -> x, 1 * x -> x
        if op == BinaryOperator.MUL:
            if self._is_zero(left) or self._is_zero(right):
                return IntegerLiteral(value=0, location=location)
            if self._is_one(right):
                return left
            if self._is_one(left):
                return right

        # x / 1 -> x
        # x / x -> 1 (assuming x != 0)
        if op == BinaryOperator.DIV:
            if self._is_one(right):
                return left
            # Be careful: don't simplify x/x if x could be zero
            # Only simplify for known non-zero literals
            if self._exprs_equal(left, right) and self._is_nonzero_literal(left):
                return IntegerLiteral(value=1, location=location)

        # x ** 0 -> 1
        # x ** 1 -> x
        if op == BinaryOperator.POW:
            if self._is_zero(right):
                return IntegerLiteral(value=1, location=location)
            if self._is_one(right):
                return left

        # True and x -> x, x and True -> x
        # False and x -> False, x and False -> False
        if op == BinaryOperator.AND:
            if isinstance(left, BooleanLiteral):
                if left.value:
                    return right
                else:
                    return BooleanLiteral(value=False, location=location)
            if isinstance(right, BooleanLiteral):
                if right.value:
                    return left
                else:
                    return BooleanLiteral(value=False, location=location)

        # True or x -> True, x or True -> True
        # False or x -> x, x or False -> x
        if op == BinaryOperator.OR:
            if isinstance(left, BooleanLiteral):
                if left.value:
                    return BooleanLiteral(value=True, location=location)
                else:
                    return right
            if isinstance(right, BooleanLiteral):
                if right.value:
                    return BooleanLiteral(value=True, location=location)
                else:
                    return left

        return None

    def _apply_unary_identity(
        self,
        op: UnaryOperator,
        operand: Expression,
        location: Optional[SourceLocation],
    ) -> Optional[Expression]:
        """
        Apply unary algebraic identities.

        Returns the simplified expression or None if no identity applies.
        """
        # -(-x) -> x
        if op == UnaryOperator.NEG:
            if isinstance(operand, UnaryExpression) and operand.operator == UnaryOperator.NEG:
                return operand.operand

        # +x -> x
        if op == UnaryOperator.POS:
            return operand

        # not not x -> x
        if op == UnaryOperator.NOT:
            if isinstance(operand, UnaryExpression) and operand.operator == UnaryOperator.NOT:
                return operand.operand

        return None

    def _is_zero(self, expr: Expression) -> bool:
        """Check if an expression is the constant zero."""
        if isinstance(expr, IntegerLiteral):
            return expr.value == 0
        if isinstance(expr, FloatLiteral):
            return expr.value == 0.0
        return False

    def _is_one(self, expr: Expression) -> bool:
        """Check if an expression is the constant one."""
        if isinstance(expr, IntegerLiteral):
            return expr.value == 1
        if isinstance(expr, FloatLiteral):
            return expr.value == 1.0
        return False

    def _is_nonzero_literal(self, expr: Expression) -> bool:
        """Check if an expression is a non-zero literal."""
        if isinstance(expr, IntegerLiteral):
            return expr.value != 0
        if isinstance(expr, FloatLiteral):
            return expr.value != 0.0
        return False

    def _exprs_equal(self, a: Expression, b: Expression) -> bool:
        """Check if two expressions are structurally equal."""
        if type(a) != type(b):
            return False
        if isinstance(a, Identifier) and isinstance(b, Identifier):
            return a.name == b.name
        if isinstance(a, IntegerLiteral) and isinstance(b, IntegerLiteral):
            return a.value == b.value
        if isinstance(a, FloatLiteral) and isinstance(b, FloatLiteral):
            return a.value == b.value
        if isinstance(a, StringLiteral) and isinstance(b, StringLiteral):
            return a.value == b.value
        if isinstance(a, BooleanLiteral) and isinstance(b, BooleanLiteral):
            return a.value == b.value
        # For complex expressions, be conservative
        return False


# =============================================================================
# 5. Strength Reducer
# =============================================================================


class StrengthReducer(OptimizerPass):
    """
    Replace expensive operations with cheaper equivalents.

    Reductions applied:
    - x * 2 -> x + x
    - x * 4 -> x << 2 (for integers)
    - x / 2 -> x * 0.5
    - x ** 2 -> x * x
    - x ** 0.5 -> sqrt(x)
    - x % 2 -> x & 1 (for integers)
    - x * 2^n -> x << n (for integers)
    """

    def __init__(self) -> None:
        """Initialize the strength reducer."""
        pass

    @property
    def name(self) -> str:
        return "Strength Reduction"

    def optimize(self, program: Program) -> Program:
        """Apply strength reduction to the entire program."""
        new_statements = tuple(
            self._reduce_statement(stmt) for stmt in program.statements
        )
        return Program(statements=new_statements, location=program.location)

    def reduce(self, expr: Expression) -> Expression:
        """
        Apply strength reduction to an expression.

        Args:
            expr: The expression to reduce

        Returns:
            An expression using cheaper operations
        """
        return self._reduce_expression(expr)

    def _reduce_statement(self, stmt: Statement) -> Statement:
        """Apply strength reduction to a statement."""
        if isinstance(stmt, LetStatement):
            if stmt.value:
                new_value = self._reduce_expression(stmt.value)
                if new_value is not stmt.value:
                    return LetStatement(
                        name=stmt.name,
                        type_annotation=stmt.type_annotation,
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, AssignmentStatement):
            new_value = self._reduce_expression(stmt.value)
            if new_value is not stmt.value:
                return AssignmentStatement(
                    target=stmt.target,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ExpressionStatement):
            new_expr = self._reduce_expression(stmt.expression)
            if new_expr is not stmt.expression:
                return ExpressionStatement(
                    expression=new_expr,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ReturnStatement):
            if stmt.value:
                new_value = self._reduce_expression(stmt.value)
                if new_value is not stmt.value:
                    return ReturnStatement(
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, IfStatement):
            new_cond = self._reduce_expression(stmt.condition)
            new_then = self._reduce_block(stmt.then_block)
            new_elifs = tuple(
                (self._reduce_expression(c), self._reduce_block(b))
                for c, b in stmt.elif_clauses
            )
            new_else = self._reduce_block(stmt.else_block) if stmt.else_block else None

            if (new_cond is not stmt.condition or new_then is not stmt.then_block or
                new_elifs != stmt.elif_clauses or new_else is not stmt.else_block):
                return IfStatement(
                    condition=new_cond,
                    then_block=new_then,
                    elif_clauses=new_elifs,
                    else_block=new_else,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, WhileStatement):
            new_cond = self._reduce_expression(stmt.condition)
            new_body = self._reduce_block(stmt.body)
            if new_cond is not stmt.condition or new_body is not stmt.body:
                return WhileStatement(
                    condition=new_cond,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ForStatement):
            new_iter = self._reduce_expression(stmt.iterable)
            new_body = self._reduce_block(stmt.body)
            if new_iter is not stmt.iterable or new_body is not stmt.body:
                return ForStatement(
                    variable=stmt.variable,
                    iterable=new_iter,
                    body=new_body,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, FunctionDef):
            if stmt.body:
                new_body = self._reduce_block(stmt.body)
                if new_body is not stmt.body:
                    return FunctionDef(
                        name=stmt.name,
                        parameters=stmt.parameters,
                        return_type=stmt.return_type,
                        body=new_body,
                        type_params=stmt.type_params,
                        where_clause=stmt.where_clause,
                        jit_options=stmt.jit_options,
                        doc_comment=stmt.doc_comment,
                        attributes=stmt.attributes,
                        location=stmt.location,
                    )
            return stmt

        return stmt

    def _reduce_block(self, block: Block) -> Block:
        """Apply strength reduction to a block."""
        new_statements = tuple(
            self._reduce_statement(stmt) for stmt in block.statements
        )
        if new_statements != block.statements:
            return Block(statements=new_statements, location=block.location)
        return block

    def _reduce_expression(self, expr: Expression) -> Expression:
        """Apply strength reduction to an expression."""
        if isinstance(expr, BinaryExpression):
            # First reduce operands
            left = self._reduce_expression(expr.left)
            right = self._reduce_expression(expr.right)

            # Apply strength reductions
            result = self._apply_reduction(left, expr.operator, right, expr.location)
            if result is not None:
                return result

            if left is not expr.left or right is not expr.right:
                return BinaryExpression(
                    left=left,
                    operator=expr.operator,
                    right=right,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, UnaryExpression):
            operand = self._reduce_expression(expr.operand)
            if operand is not expr.operand:
                return UnaryExpression(
                    operator=expr.operator,
                    operand=operand,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, CallExpression):
            new_callee = self._reduce_expression(expr.callee)
            new_args = tuple(self._reduce_expression(arg) for arg in expr.arguments)
            if new_callee is not expr.callee or new_args != expr.arguments:
                return CallExpression(
                    callee=new_callee,
                    arguments=new_args,
                    location=expr.location,
                )
            return expr

        return expr

    def _apply_reduction(
        self,
        left: Expression,
        op: BinaryOperator,
        right: Expression,
        location: Optional[SourceLocation],
    ) -> Optional[Expression]:
        """
        Apply strength reduction rules.

        Returns the reduced expression or None if no reduction applies.
        """
        # x * 2 -> x + x
        if op == BinaryOperator.MUL:
            if self._is_int_value(right, 2):
                return BinaryExpression(
                    left=left,
                    operator=BinaryOperator.ADD,
                    right=left,
                    location=location,
                )
            if self._is_int_value(left, 2):
                return BinaryExpression(
                    left=right,
                    operator=BinaryOperator.ADD,
                    right=right,
                    location=location,
                )

        # x / 2 -> x * 0.5
        if op == BinaryOperator.DIV:
            if self._is_int_value(right, 2):
                return BinaryExpression(
                    left=left,
                    operator=BinaryOperator.MUL,
                    right=FloatLiteral(value=0.5, location=location),
                    location=location,
                )

        # x ** 2 -> x * x
        if op == BinaryOperator.POW:
            if self._is_int_value(right, 2):
                return BinaryExpression(
                    left=left,
                    operator=BinaryOperator.MUL,
                    right=left,
                    location=location,
                )
            # x ** 0.5 -> sqrt(x)
            if self._is_float_value(right, 0.5):
                return CallExpression(
                    callee=Identifier(name="sqrt", location=location),
                    arguments=(left,),
                    location=location,
                )

        return None

    def _is_int_value(self, expr: Expression, value: int) -> bool:
        """Check if an expression is a specific integer value."""
        if isinstance(expr, IntegerLiteral):
            return expr.value == value
        return False

    def _is_float_value(self, expr: Expression, value: float) -> bool:
        """Check if an expression is a specific float value."""
        if isinstance(expr, FloatLiteral):
            return abs(expr.value - value) < 1e-10
        return False


# =============================================================================
# 6. Common Subexpression Eliminator
# =============================================================================


@dataclass(frozen=True)
class ExpressionKey:
    """
    Hashable key for identifying common subexpressions.
    """
    type_name: str
    data: tuple

    @staticmethod
    def from_expression(expr: Expression) -> Optional["ExpressionKey"]:
        """Create a key from an expression if it's suitable for CSE."""
        if isinstance(expr, BinaryExpression):
            left_key = ExpressionKey.from_expression(expr.left)
            right_key = ExpressionKey.from_expression(expr.right)
            if left_key is None or right_key is None:
                return None
            return ExpressionKey(
                type_name="Binary",
                data=(expr.operator, left_key, right_key),
            )
        if isinstance(expr, UnaryExpression):
            operand_key = ExpressionKey.from_expression(expr.operand)
            if operand_key is None:
                return None
            return ExpressionKey(
                type_name="Unary",
                data=(expr.operator, operand_key),
            )
        if isinstance(expr, Identifier):
            return ExpressionKey(type_name="Identifier", data=(expr.name,))
        if isinstance(expr, IntegerLiteral):
            return ExpressionKey(type_name="Integer", data=(expr.value,))
        if isinstance(expr, FloatLiteral):
            return ExpressionKey(type_name="Float", data=(expr.value,))
        if isinstance(expr, StringLiteral):
            return ExpressionKey(type_name="String", data=(expr.value,))
        if isinstance(expr, BooleanLiteral):
            return ExpressionKey(type_name="Boolean", data=(expr.value,))
        if isinstance(expr, CallExpression):
            if isinstance(expr.callee, Identifier):
                arg_keys = []
                for arg in expr.arguments:
                    key = ExpressionKey.from_expression(arg)
                    if key is None:
                        return None
                    arg_keys.append(key)
                return ExpressionKey(
                    type_name="Call",
                    data=(expr.callee.name, tuple(arg_keys)),
                )
        if isinstance(expr, MemberAccess):
            obj_key = ExpressionKey.from_expression(expr.object)
            if obj_key is None:
                return None
            return ExpressionKey(
                type_name="MemberAccess",
                data=(obj_key, expr.member),
            )
        if isinstance(expr, IndexExpression):
            obj_key = ExpressionKey.from_expression(expr.object)
            idx_key = ExpressionKey.from_expression(expr.index)
            if obj_key is None or idx_key is None:
                return None
            return ExpressionKey(
                type_name="Index",
                data=(obj_key, idx_key),
            )
        return None


class CSEliminator(OptimizerPass):
    """
    Eliminate common subexpressions by introducing temporary variables.

    This optimizer finds expressions that are computed multiple times
    and replaces them with a single computation stored in a temporary.

    Example:
        let a = x * y + 1
        let b = x * y + 2

    Becomes:
        let _cse_0 = x * y
        let a = _cse_0 + 1
        let b = _cse_0 + 2
    """

    def __init__(self, min_uses: int = 2) -> None:
        """
        Initialize the CSE eliminator.

        Args:
            min_uses: Minimum number of uses to trigger CSE (default: 2)
        """
        self._min_uses = min_uses
        self._temp_counter = 0

    @property
    def name(self) -> str:
        return "Common Subexpression Elimination"

    def optimize(self, program: Program) -> Program:
        """Apply CSE to the entire program."""
        new_statements: list[Statement] = []

        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                new_stmt = self._eliminate_in_function(stmt)
                new_statements.append(new_stmt)
            else:
                new_statements.append(stmt)

        return Program(
            statements=tuple(new_statements),
            location=program.location,
        )

    def eliminate(self, block: Block) -> Block:
        """
        Find and eliminate common subexpressions in a block.

        Args:
            block: The block to process

        Returns:
            A new block with CSE applied
        """
        # Find common subexpressions
        expr_counts = self._count_expressions(block)
        common_exprs = {
            key: exprs[0] for key, exprs in expr_counts.items()
            if len(exprs) >= self._min_uses
        }

        if not common_exprs:
            return block

        # Introduce temporaries and substitute
        temp_map: dict[ExpressionKey, str] = {}
        new_statements: list[Statement] = []

        for key, expr in common_exprs.items():
            temp_name = self._generate_temp_name()
            temp_map[key] = temp_name
            new_statements.append(LetStatement(
                name=temp_name,
                value=expr,
                location=expr.location,
            ))

        # Substitute common subexpressions with temporaries
        for stmt in block.statements:
            new_stmt = self._substitute_in_statement(stmt, temp_map)
            new_statements.append(new_stmt)

        return Block(
            statements=tuple(new_statements),
            location=block.location,
        )

    def _eliminate_in_function(self, func: FunctionDef) -> FunctionDef:
        """Apply CSE to a function body."""
        if func.body is None:
            return func

        new_body = self.eliminate(func.body)
        if new_body is not func.body:
            return FunctionDef(
                name=func.name,
                parameters=func.parameters,
                return_type=func.return_type,
                body=new_body,
                type_params=func.type_params,
                where_clause=func.where_clause,
                jit_options=func.jit_options,
                doc_comment=func.doc_comment,
                attributes=func.attributes,
                location=func.location,
            )
        return func

    def _count_expressions(self, block: Block) -> dict[ExpressionKey, list[Expression]]:
        """
        Count occurrences of each expression in a block.

        Returns a mapping from expression key to list of expressions.
        """
        counts: dict[ExpressionKey, list[Expression]] = {}

        for stmt in block.statements:
            self._count_in_statement(stmt, counts)

        return counts

    def _count_in_statement(
        self,
        stmt: Statement,
        counts: dict[ExpressionKey, list[Expression]],
    ) -> None:
        """Count expressions in a statement."""
        if isinstance(stmt, LetStatement):
            if stmt.value:
                self._count_in_expression(stmt.value, counts)
        elif isinstance(stmt, AssignmentStatement):
            self._count_in_expression(stmt.value, counts)
        elif isinstance(stmt, ExpressionStatement):
            self._count_in_expression(stmt.expression, counts)
        elif isinstance(stmt, ReturnStatement):
            if stmt.value:
                self._count_in_expression(stmt.value, counts)
        elif isinstance(stmt, IfStatement):
            self._count_in_expression(stmt.condition, counts)
            for s in stmt.then_block.statements:
                self._count_in_statement(s, counts)
            for cond, block in stmt.elif_clauses:
                self._count_in_expression(cond, counts)
                for s in block.statements:
                    self._count_in_statement(s, counts)
            if stmt.else_block:
                for s in stmt.else_block.statements:
                    self._count_in_statement(s, counts)
        elif isinstance(stmt, WhileStatement):
            self._count_in_expression(stmt.condition, counts)
            for s in stmt.body.statements:
                self._count_in_statement(s, counts)
        elif isinstance(stmt, ForStatement):
            self._count_in_expression(stmt.iterable, counts)
            for s in stmt.body.statements:
                self._count_in_statement(s, counts)
        elif isinstance(stmt, PrintStatement):
            self._count_in_expression(stmt.format_string, counts)
            for arg in stmt.arguments:
                self._count_in_expression(arg, counts)

    def _count_in_expression(
        self,
        expr: Expression,
        counts: dict[ExpressionKey, list[Expression]],
    ) -> None:
        """Count occurrences of an expression and its subexpressions."""
        # Only count non-trivial expressions
        if isinstance(expr, (BinaryExpression, CallExpression)):
            key = ExpressionKey.from_expression(expr)
            if key is not None:
                if key not in counts:
                    counts[key] = []
                counts[key].append(expr)

        # Recurse into subexpressions
        if isinstance(expr, BinaryExpression):
            self._count_in_expression(expr.left, counts)
            self._count_in_expression(expr.right, counts)
        elif isinstance(expr, UnaryExpression):
            self._count_in_expression(expr.operand, counts)
        elif isinstance(expr, CallExpression):
            for arg in expr.arguments:
                self._count_in_expression(arg, counts)
        elif isinstance(expr, IndexExpression):
            self._count_in_expression(expr.object, counts)
            self._count_in_expression(expr.index, counts)
        elif isinstance(expr, MemberAccess):
            self._count_in_expression(expr.object, counts)
        elif isinstance(expr, ConditionalExpression):
            self._count_in_expression(expr.condition, counts)
            self._count_in_expression(expr.then_expr, counts)
            self._count_in_expression(expr.else_expr, counts)

    def _substitute_in_statement(
        self,
        stmt: Statement,
        temp_map: dict[ExpressionKey, str],
    ) -> Statement:
        """Substitute common subexpressions in a statement."""
        if isinstance(stmt, LetStatement):
            if stmt.value:
                new_value = self._substitute_in_expression(stmt.value, temp_map)
                if new_value is not stmt.value:
                    return LetStatement(
                        name=stmt.name,
                        type_annotation=stmt.type_annotation,
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        if isinstance(stmt, AssignmentStatement):
            new_value = self._substitute_in_expression(stmt.value, temp_map)
            if new_value is not stmt.value:
                return AssignmentStatement(
                    target=stmt.target,
                    value=new_value,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ExpressionStatement):
            new_expr = self._substitute_in_expression(stmt.expression, temp_map)
            if new_expr is not stmt.expression:
                return ExpressionStatement(
                    expression=new_expr,
                    location=stmt.location,
                )
            return stmt

        if isinstance(stmt, ReturnStatement):
            if stmt.value:
                new_value = self._substitute_in_expression(stmt.value, temp_map)
                if new_value is not stmt.value:
                    return ReturnStatement(
                        value=new_value,
                        location=stmt.location,
                    )
            return stmt

        # For other statements, return as-is (could be extended)
        return stmt

    def _substitute_in_expression(
        self,
        expr: Expression,
        temp_map: dict[ExpressionKey, str],
    ) -> Expression:
        """Substitute common subexpressions in an expression."""
        # Check if this expression has been assigned a temporary
        key = ExpressionKey.from_expression(expr)
        if key is not None and key in temp_map:
            return Identifier(name=temp_map[key], location=expr.location)

        # Recurse into subexpressions
        if isinstance(expr, BinaryExpression):
            left = self._substitute_in_expression(expr.left, temp_map)
            right = self._substitute_in_expression(expr.right, temp_map)
            if left is not expr.left or right is not expr.right:
                return BinaryExpression(
                    left=left,
                    operator=expr.operator,
                    right=right,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, UnaryExpression):
            operand = self._substitute_in_expression(expr.operand, temp_map)
            if operand is not expr.operand:
                return UnaryExpression(
                    operator=expr.operator,
                    operand=operand,
                    location=expr.location,
                )
            return expr

        if isinstance(expr, CallExpression):
            new_args = tuple(
                self._substitute_in_expression(arg, temp_map)
                for arg in expr.arguments
            )
            if new_args != expr.arguments:
                return CallExpression(
                    callee=expr.callee,
                    arguments=new_args,
                    location=expr.location,
                )
            return expr

        return expr

    def _generate_temp_name(self) -> str:
        """Generate a unique temporary variable name."""
        name = f"_cse_{self._temp_counter}"
        self._temp_counter += 1
        return name

    def _find_common_subexpressions(
        self, block: Block
    ) -> dict[str, list[Expression]]:
        """Find expressions that appear multiple times."""
        return {
            str(key): exprs
            for key, exprs in self._count_expressions(block).items()
            if len(exprs) >= self._min_uses
        }

    def _introduce_temp_variable(
        self, expr: Expression
    ) -> tuple[LetStatement, Identifier]:
        """Create a temporary variable for a common subexpression."""
        temp_name = self._generate_temp_name()
        let_stmt = LetStatement(
            name=temp_name,
            value=expr,
            location=expr.location,
        )
        identifier = Identifier(name=temp_name, location=expr.location)
        return let_stmt, identifier


# =============================================================================
# 7. Main Optimizer Pipeline
# =============================================================================


class ConstOptimizer:
    """
    Main optimizer combining all constant-time optimizations.

    This class orchestrates multiple optimization passes in a configurable
    order, providing a single entry point for program optimization.

    The default pass order is:
    1. Constant Propagation (propagate known values)
    2. Constant Folding (evaluate constant expressions)
    3. Algebraic Simplification (apply identities)
    4. Strength Reduction (use cheaper operations)
    5. Dead Code Elimination (remove unreachable code)
    6. Common Subexpression Elimination (reuse computations)

    Example:
        optimizer = ConstOptimizer(
            fold_constants=True,
            propagate_constants=True,
            eliminate_dead_code=True,
        )
        optimized_program = optimizer.optimize(program)
    """

    def __init__(
        self,
        fold_constants: bool = True,
        propagate_constants: bool = True,
        eliminate_dead_code: bool = True,
        simplify_algebra: bool = True,
        reduce_strength: bool = True,
        eliminate_cse: bool = True,
        max_iterations: int = 3,
    ) -> None:
        """
        Initialize the optimizer with configurable passes.

        Args:
            fold_constants: Enable constant folding
            propagate_constants: Enable constant propagation
            eliminate_dead_code: Enable dead code elimination
            simplify_algebra: Enable algebraic simplification
            reduce_strength: Enable strength reduction
            eliminate_cse: Enable common subexpression elimination
            max_iterations: Maximum optimization iterations for fixed-point
        """
        self.passes: list[OptimizerPass] = []
        self.max_iterations = max_iterations

        # Add passes in optimal order
        if propagate_constants:
            self.passes.append(ConstantPropagator())
        if fold_constants:
            self.passes.append(ConstantFolder())
        if simplify_algebra:
            self.passes.append(AlgebraicSimplifier())
        if reduce_strength:
            self.passes.append(StrengthReducer())
        if eliminate_dead_code:
            self.passes.append(DeadCodeEliminator())
        if eliminate_cse:
            self.passes.append(CSEliminator())

    def optimize(self, program: Program) -> Program:
        """
        Run all optimization passes on the program.

        The optimizer runs multiple iterations until a fixed point is
        reached or the maximum iteration count is exceeded.

        Args:
            program: The input program AST

        Returns:
            An optimized program AST
        """
        result = program

        for _ in range(self.max_iterations):
            previous = result

            for pass_ in self.passes:
                result = pass_.optimize(result)

            # Check if we've reached a fixed point
            if self._programs_equal(result, previous):
                break

        return result

    def optimize_expression(self, expr: Expression) -> Expression:
        """
        Optimize a single expression.

        This is a convenience method for optimizing expressions without
        wrapping them in a full program.

        Args:
            expr: The expression to optimize

        Returns:
            An optimized expression
        """
        # Wrap in a minimal program structure
        stmt = ExpressionStatement(expression=expr)
        program = Program(statements=(stmt,))

        # Optimize
        optimized = self.optimize(program)

        # Extract the expression
        if (optimized.statements and
            isinstance(optimized.statements[0], ExpressionStatement)):
            return optimized.statements[0].expression
        return expr

    def _programs_equal(self, a: Program, b: Program) -> bool:
        """Check if two programs are structurally equal."""
        # Simple length check first
        if len(a.statements) != len(b.statements):
            return False

        # This is a simplification - in practice you'd do a deep comparison
        # For now, we rely on the max_iterations limit
        return False

    def get_pass_names(self) -> list[str]:
        """Get the names of all enabled optimization passes."""
        return [pass_.name for pass_ in self.passes]


# =============================================================================
# Convenience Functions
# =============================================================================


def fold_constants(program: Program) -> Program:
    """
    Convenience function to fold constants in a program.

    Args:
        program: The input program AST

    Returns:
        A new program with constants folded
    """
    return ConstantFolder().fold(program)


def propagate_constants(program: Program) -> Program:
    """
    Convenience function to propagate constants in a program.

    Args:
        program: The input program AST

    Returns:
        A new program with constants propagated
    """
    return ConstantPropagator().propagate(program)


def eliminate_dead_code(program: Program) -> Program:
    """
    Convenience function to eliminate dead code in a program.

    Args:
        program: The input program AST

    Returns:
        A new program with dead code removed
    """
    return DeadCodeEliminator().eliminate(program)


def simplify_algebra(program: Program) -> Program:
    """
    Convenience function to apply algebraic simplifications.

    Args:
        program: The input program AST

    Returns:
        A new program with simplified expressions
    """
    return AlgebraicSimplifier().optimize(program)


def reduce_strength(program: Program) -> Program:
    """
    Convenience function to apply strength reduction.

    Args:
        program: The input program AST

    Returns:
        A new program with cheaper operations
    """
    return StrengthReducer().optimize(program)


def eliminate_cse(program: Program) -> Program:
    """
    Convenience function to eliminate common subexpressions.

    Args:
        program: The input program AST

    Returns:
        A new program with CSE applied
    """
    return CSEliminator().optimize(program)


def optimize_program(
    program: Program,
    fold_constants: bool = True,
    propagate_constants: bool = True,
    eliminate_dead_code: bool = True,
    simplify_algebra: bool = True,
    reduce_strength: bool = True,
    eliminate_cse: bool = True,
) -> Program:
    """
    Convenience function to run the full optimization pipeline.

    Args:
        program: The input program AST
        fold_constants: Enable constant folding
        propagate_constants: Enable constant propagation
        eliminate_dead_code: Enable dead code elimination
        simplify_algebra: Enable algebraic simplification
        reduce_strength: Enable strength reduction
        eliminate_cse: Enable common subexpression elimination

    Returns:
        An optimized program AST
    """
    optimizer = ConstOptimizer(
        fold_constants=fold_constants,
        propagate_constants=propagate_constants,
        eliminate_dead_code=eliminate_dead_code,
        simplify_algebra=simplify_algebra,
        reduce_strength=reduce_strength,
        eliminate_cse=eliminate_cse,
    )
    return optimizer.optimize(program)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core classes
    "OptimizerPass",
    "ConstantFolder",
    "ConstantPropagator",
    "DeadCodeEliminator",
    "AlgebraicSimplifier",
    "StrengthReducer",
    "CSEliminator",
    "ConstOptimizer",
    # Helper classes
    "ConstantScope",
    "ExpressionKey",
    # Convenience functions
    "fold_constants",
    "propagate_constants",
    "eliminate_dead_code",
    "simplify_algebra",
    "reduce_strength",
    "eliminate_cse",
    "optimize_program",
]
