"""
MathViz Code Formatter.

Provides AST-based code formatting for MathViz source files,
ensuring consistent style and indentation.

Usage:
    mathviz fmt input.mviz
    mathviz fmt --check .
    mathviz fmt --diff .
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from mathviz.compiler.ast_nodes import (
    AssignmentStatement,
    BinaryExpression,
    BinaryOperator,
    Block,
    BooleanLiteral,
    BreakStatement,
    CallExpression,
    ClassDef,
    CompoundAssignment,
    ConditionalExpression,
    ConstructorPattern,
    ContinueStatement,
    DictLiteral,
    EnumDef,
    EnumPattern,
    EnumVariant,
    EnumVariantAccess,
    ErrExpression,
    # Expressions
    Expression,
    ExpressionStatement,
    FloatLiteral,
    ForStatement,
    FunctionDef,
    FunctionType,
    GenericType,
    Identifier,
    IdentifierPattern,
    IfStatement,
    ImplBlock,
    ImportStatement,
    IndexExpression,
    IntegerLiteral,
    JitMode,
    JitOptions,
    LambdaExpression,
    LetStatement,
    ListLiteral,
    LiteralPattern,
    MatchExpression,
    MemberAccess,
    Method,
    ModuleDecl,
    NoneLiteral,
    OkExpression,
    Parameter,
    PassStatement,
    Pattern,
    PlayStatement,
    PrintStatement,
    # Program
    Program,
    RangeExpression,
    ReturnStatement,
    SceneDef,
    SelfExpression,
    SetLiteral,
    SimpleType,
    SomeExpression,
    # Statements
    Statement,
    StringLiteral,
    # OOP
    StructDef,
    StructLiteral,
    TraitDef,
    TraitMethod,
    TupleLiteral,
    TuplePattern,
    # Types
    TypeAnnotation,
    UnaryExpression,
    UnaryOperator,
    UnwrapExpression,
    UseStatement,
    Visibility,
    WaitStatement,
    WhileStatement,
)
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser

# =============================================================================
# Formatter Configuration
# =============================================================================


@dataclass
class FormatConfig:
    """Configuration for the code formatter."""

    indent_size: int = 4
    max_line_length: int = 100
    use_spaces: bool = True
    trailing_newline: bool = True
    blank_lines_between_functions: int = 2
    blank_lines_between_top_level: int = 1
    space_around_operators: bool = True
    space_after_comma: bool = True
    space_after_colon: bool = True
    break_long_lines: bool = True


# =============================================================================
# Binary Operator Mapping
# =============================================================================


BINARY_OP_SYMBOLS: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "+",
    BinaryOperator.SUB: "-",
    BinaryOperator.MUL: "*",
    BinaryOperator.DIV: "/",
    BinaryOperator.FLOOR_DIV: "//",
    BinaryOperator.MOD: "%",
    BinaryOperator.POW: "^",
    BinaryOperator.EQ: "==",
    BinaryOperator.NE: "!=",
    BinaryOperator.LT: "<",
    BinaryOperator.GT: ">",
    BinaryOperator.LE: "<=",
    BinaryOperator.GE: ">=",
    BinaryOperator.AND: "and",
    BinaryOperator.OR: "or",
    BinaryOperator.IN: "in",
    BinaryOperator.NOT_IN: "not in",
    BinaryOperator.ELEMENT_OF: "in",
    BinaryOperator.NOT_ELEMENT_OF: "not in",
    BinaryOperator.SUBSET: "<=",
    BinaryOperator.SUPERSET: ">=",
    BinaryOperator.PROPER_SUBSET: "<",
    BinaryOperator.PROPER_SUPERSET: ">",
    BinaryOperator.UNION: "|",
    BinaryOperator.INTERSECTION: "&",
    BinaryOperator.SET_DIFF: "-",
}


UNARY_OP_SYMBOLS: dict[UnaryOperator, str] = {
    UnaryOperator.NEG: "-",
    UnaryOperator.NOT: "not ",
    UnaryOperator.POS: "+",
}


COMPOUND_OP_SYMBOLS: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "+=",
    BinaryOperator.SUB: "-=",
    BinaryOperator.MUL: "*=",
    BinaryOperator.DIV: "/=",
}


# =============================================================================
# Code Formatter
# =============================================================================


class Formatter:
    """
    AST-based code formatter for MathViz.

    Traverses the AST and produces consistently formatted source code.
    """

    def __init__(self, config: FormatConfig | None = None) -> None:
        """Initialize the formatter with optional configuration."""
        self.config = config or FormatConfig()
        self._indent_level = 0
        self._output_lines: list[str] = []
        self._current_line = ""

    def format_program(self, program: Program) -> str:
        """Format a complete program."""
        self._output_lines = []
        self._current_line = ""
        self._indent_level = 0

        prev_stmt_type: type | None = None

        for i, stmt in enumerate(program.statements):
            # Add blank lines between certain statement types
            if i > 0:
                if self._should_add_blank_lines(prev_stmt_type, type(stmt)):
                    self._emit_blank_lines(self.config.blank_lines_between_functions)
                elif prev_stmt_type != type(stmt):
                    self._emit_blank_lines(self.config.blank_lines_between_top_level)

            self._format_statement(stmt)
            prev_stmt_type = type(stmt)

        # Join lines and handle trailing newline
        result = "\n".join(self._output_lines)
        if self.config.trailing_newline and result and not result.endswith("\n"):
            result += "\n"

        return result

    def _should_add_blank_lines(self, prev: type | None, curr: type) -> bool:
        """Determine if blank lines should be added between statements."""
        function_types = (FunctionDef, ClassDef, SceneDef, StructDef, ImplBlock, TraitDef, EnumDef)
        return prev in function_types or curr in function_types

    def _emit_blank_lines(self, count: int) -> None:
        """Emit the specified number of blank lines."""
        for _ in range(count):
            self._output_lines.append("")

    def _indent(self) -> str:
        """Get the current indentation string."""
        char = " " if self.config.use_spaces else "\t"
        return char * (self._indent_level * self.config.indent_size)

    def _emit_line(self, line: str = "") -> None:
        """Emit a line of output."""
        if line:
            self._output_lines.append(self._indent() + line)
        else:
            self._output_lines.append("")

    # -------------------------------------------------------------------------
    # Statement Formatting
    # -------------------------------------------------------------------------

    def _format_statement(self, stmt: Statement) -> None:
        """Format a single statement."""
        if isinstance(stmt, LetStatement):
            self._format_let_statement(stmt)
        elif isinstance(stmt, FunctionDef):
            self._format_function_def(stmt)
        elif isinstance(stmt, ClassDef):
            self._format_class_def(stmt)
        elif isinstance(stmt, SceneDef):
            self._format_scene_def(stmt)
        elif isinstance(stmt, IfStatement):
            self._format_if_statement(stmt)
        elif isinstance(stmt, ForStatement):
            self._format_for_statement(stmt)
        elif isinstance(stmt, WhileStatement):
            self._format_while_statement(stmt)
        elif isinstance(stmt, ReturnStatement):
            self._format_return_statement(stmt)
        elif isinstance(stmt, BreakStatement):
            self._emit_line("break")
        elif isinstance(stmt, ContinueStatement):
            self._emit_line("continue")
        elif isinstance(stmt, PassStatement):
            self._emit_line("pass")
        elif isinstance(stmt, ImportStatement):
            self._format_import_statement(stmt)
        elif isinstance(stmt, UseStatement):
            self._format_use_statement(stmt)
        elif isinstance(stmt, PrintStatement):
            self._format_print_statement(stmt)
        elif isinstance(stmt, PlayStatement):
            self._format_play_statement(stmt)
        elif isinstance(stmt, WaitStatement):
            self._format_wait_statement(stmt)
        elif isinstance(stmt, ModuleDecl):
            self._format_module_decl(stmt)
        elif isinstance(stmt, AssignmentStatement):
            self._format_assignment_statement(stmt)
        elif isinstance(stmt, CompoundAssignment):
            self._format_compound_assignment(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self._format_expression_statement(stmt)
        elif isinstance(stmt, StructDef):
            self._format_struct_def(stmt)
        elif isinstance(stmt, ImplBlock):
            self._format_impl_block(stmt)
        elif isinstance(stmt, TraitDef):
            self._format_trait_def(stmt)
        elif isinstance(stmt, EnumDef):
            self._format_enum_def(stmt)

    def _format_let_statement(self, stmt: LetStatement) -> None:
        """Format a let statement."""
        parts = [f"let {stmt.name}"]

        if stmt.type_annotation:
            parts.append(f": {self._format_type_annotation(stmt.type_annotation)}")

        if stmt.value:
            parts.append(f" = {self._format_expression(stmt.value)}")

        self._emit_line("".join(parts))

    def _format_function_def(self, stmt: FunctionDef) -> None:
        """Format a function definition."""
        # JIT decorator
        if stmt.jit_options and stmt.jit_options.mode != JitMode.NONE:
            decorator = self._format_jit_decorator(stmt.jit_options)
            self._emit_line(decorator)

        # Function signature
        params = ", ".join(self._format_parameter(p) for p in stmt.parameters)
        line = f"fn {stmt.name}({params})"

        if stmt.return_type:
            line += f" -> {self._format_type_annotation(stmt.return_type)}"

        if stmt.body:
            line += " {"
            self._emit_line(line)
            self._format_block_contents(stmt.body)
            self._emit_line("}")
        else:
            self._emit_line(line)

    def _format_jit_decorator(self, options: JitOptions) -> str:
        """Format a JIT decorator."""
        mode_str = options.mode.value
        args = options.to_decorator_args()
        if args:
            return f"@{mode_str}({args})"
        return f"@{mode_str}"

    def _format_class_def(self, stmt: ClassDef) -> None:
        """Format a class definition."""
        line = f"class {stmt.name}"

        if stmt.base_classes:
            bases = ", ".join(stmt.base_classes)
            line += f"({bases})"

        if stmt.body:
            line += " {"
            self._emit_line(line)
            self._format_block_contents(stmt.body)
            self._emit_line("}")
        else:
            self._emit_line(line)

    def _format_scene_def(self, stmt: SceneDef) -> None:
        """Format a scene definition."""
        line = f"scene {stmt.name}"

        if stmt.body:
            line += " {"
            self._emit_line(line)
            self._format_block_contents(stmt.body)
            self._emit_line("}")
        else:
            self._emit_line(line)

    def _format_if_statement(self, stmt: IfStatement) -> None:
        """Format an if statement."""
        cond = self._format_expression(stmt.condition)
        self._emit_line(f"if {cond} {{")
        self._format_block_contents(stmt.then_block)

        for elif_cond, elif_block in stmt.elif_clauses:
            elif_cond_str = self._format_expression(elif_cond)
            self._emit_line(f"}} elif {elif_cond_str} {{")
            self._format_block_contents(elif_block)

        if stmt.else_block:
            self._emit_line("} else {")
            self._format_block_contents(stmt.else_block)

        self._emit_line("}")

    def _format_for_statement(self, stmt: ForStatement) -> None:
        """Format a for statement."""
        iterable = self._format_expression(stmt.iterable)
        self._emit_line(f"for {stmt.variable} in {iterable} {{")
        self._format_block_contents(stmt.body)
        self._emit_line("}")

    def _format_while_statement(self, stmt: WhileStatement) -> None:
        """Format a while statement."""
        cond = self._format_expression(stmt.condition)
        self._emit_line(f"while {cond} {{")
        self._format_block_contents(stmt.body)
        self._emit_line("}")

    def _format_return_statement(self, stmt: ReturnStatement) -> None:
        """Format a return statement."""
        if stmt.value:
            value = self._format_expression(stmt.value)
            self._emit_line(f"return {value}")
        else:
            self._emit_line("return")

    def _format_import_statement(self, stmt: ImportStatement) -> None:
        """Format an import statement."""
        if stmt.is_from_import:
            names = ", ".join(f"{name} as {alias}" if alias else name for name, alias in stmt.names)
            self._emit_line(f"from {stmt.module} import {names}")
        else:
            if stmt.alias:
                self._emit_line(f"import {stmt.module} as {stmt.alias}")
            else:
                self._emit_line(f"import {stmt.module}")

    def _format_use_statement(self, stmt: UseStatement) -> None:
        """Format a use statement."""
        path = ".".join(stmt.module_path)
        if stmt.wildcard:
            path += ".*"
        if stmt.alias:
            self._emit_line(f"use {path} as {stmt.alias}")
        else:
            self._emit_line(f"use {path}")

    def _format_print_statement(self, stmt: PrintStatement) -> None:
        """Format a print statement."""
        keyword = "println" if stmt.newline else "print"
        fmt_str = self._format_expression(stmt.format_string)

        if stmt.arguments:
            args = ", ".join(self._format_expression(a) for a in stmt.arguments)
            self._emit_line(f"{keyword}({fmt_str}, {args})")
        else:
            self._emit_line(f"{keyword}({fmt_str})")

    def _format_play_statement(self, stmt: PlayStatement) -> None:
        """Format a play statement."""
        animation = self._format_expression(stmt.animation)
        if stmt.run_time:
            run_time = self._format_expression(stmt.run_time)
            self._emit_line(f"play({animation}, run_time={run_time})")
        else:
            self._emit_line(f"play({animation})")

    def _format_wait_statement(self, stmt: WaitStatement) -> None:
        """Format a wait statement."""
        if stmt.duration:
            duration = self._format_expression(stmt.duration)
            self._emit_line(f"wait({duration})")
        else:
            self._emit_line("wait()")

    def _format_module_decl(self, stmt: ModuleDecl) -> None:
        """Format a module declaration."""
        pub = "pub " if stmt.is_public else ""
        self._emit_line(f"{pub}mod {stmt.name} {{")
        self._format_block_contents(stmt.body)
        self._emit_line("}")

    def _format_assignment_statement(self, stmt: AssignmentStatement) -> None:
        """Format an assignment statement."""
        target = self._format_expression(stmt.target)
        value = self._format_expression(stmt.value)
        self._emit_line(f"{target} = {value}")

    def _format_compound_assignment(self, stmt: CompoundAssignment) -> None:
        """Format a compound assignment statement."""
        target = self._format_expression(stmt.target)
        value = self._format_expression(stmt.value)
        op = COMPOUND_OP_SYMBOLS.get(stmt.operator, "?=")
        self._emit_line(f"{target} {op} {value}")

    def _format_expression_statement(self, stmt: ExpressionStatement) -> None:
        """Format an expression statement."""
        expr = self._format_expression(stmt.expression)
        self._emit_line(expr)

    def _format_struct_def(self, stmt: StructDef) -> None:
        """Format a struct definition."""
        self._emit_line(f"struct {stmt.name} {{")
        self._indent_level += 1

        for field in stmt.fields:
            pub = "pub " if field.visibility == Visibility.PUBLIC else ""
            type_str = self._format_type_annotation(field.type_annotation)
            self._emit_line(f"{pub}{field.name}: {type_str}")

        self._indent_level -= 1
        self._emit_line("}")

    def _format_impl_block(self, stmt: ImplBlock) -> None:
        """Format an impl block."""
        if stmt.trait_name:
            self._emit_line(f"impl {stmt.trait_name} for {stmt.target_type} {{")
        else:
            self._emit_line(f"impl {stmt.target_type} {{")

        self._indent_level += 1

        for i, method in enumerate(stmt.methods):
            if i > 0:
                self._emit_blank_lines(1)
            self._format_method(method)

        self._indent_level -= 1
        self._emit_line("}")

    def _format_method(self, method: Method) -> None:
        """Format a method definition."""
        pub = "pub " if method.visibility == Visibility.PUBLIC else ""

        params = []
        if method.has_self:
            params.append("self")
        params.extend(self._format_parameter(p) for p in method.parameters)

        params_str = ", ".join(params)
        line = f"{pub}fn {method.name}({params_str})"

        if method.return_type:
            line += f" -> {self._format_type_annotation(method.return_type)}"

        if method.body:
            line += " {"
            self._emit_line(line)
            self._format_block_contents(method.body)
            self._emit_line("}")
        else:
            self._emit_line(line)

    def _format_trait_def(self, stmt: TraitDef) -> None:
        """Format a trait definition."""
        self._emit_line(f"trait {stmt.name} {{")
        self._indent_level += 1

        for i, method in enumerate(stmt.methods):
            if i > 0:
                self._emit_blank_lines(1)
            self._format_trait_method(method)

        self._indent_level -= 1
        self._emit_line("}")

    def _format_trait_method(self, method: TraitMethod) -> None:
        """Format a trait method signature."""
        params = []
        if method.has_self:
            params.append("self")
        params.extend(self._format_parameter(p) for p in method.parameters)

        params_str = ", ".join(params)
        line = f"fn {method.name}({params_str})"

        if method.return_type:
            line += f" -> {self._format_type_annotation(method.return_type)}"

        if method.has_default_impl and method.default_body:
            line += " {"
            self._emit_line(line)
            self._format_block_contents(method.default_body)
            self._emit_line("}")
        else:
            self._emit_line(line)

    def _format_enum_def(self, stmt: EnumDef) -> None:
        """Format an enum definition."""
        self._emit_line(f"enum {stmt.name} {{")
        self._indent_level += 1

        for variant in stmt.variants:
            self._format_enum_variant(variant)

        self._indent_level -= 1
        self._emit_line("}")

    def _format_enum_variant(self, variant: EnumVariant) -> None:
        """Format an enum variant."""
        if variant.fields:
            fields = ", ".join(self._format_type_annotation(f) for f in variant.fields)
            self._emit_line(f"{variant.name}({fields})")
        else:
            self._emit_line(variant.name)

    def _format_block_contents(self, block: Block) -> None:
        """Format the contents of a block."""
        self._indent_level += 1
        for stmt in block.statements:
            self._format_statement(stmt)
        self._indent_level -= 1

    # -------------------------------------------------------------------------
    # Expression Formatting
    # -------------------------------------------------------------------------

    def _format_expression(self, expr: Expression) -> str:
        """Format an expression to a string."""
        if isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, (IntegerLiteral, FloatLiteral)):
            return str(expr.value)
        elif isinstance(expr, StringLiteral):
            return f'"{expr.value}"'
        elif isinstance(expr, BooleanLiteral):
            return "true" if expr.value else "false"
        elif isinstance(expr, NoneLiteral):
            return "None"
        elif isinstance(expr, ListLiteral):
            elements = ", ".join(self._format_expression(e) for e in expr.elements)
            return f"[{elements}]"
        elif isinstance(expr, SetLiteral):
            if not expr.elements:
                return "{}"
            elements = ", ".join(self._format_expression(e) for e in expr.elements)
            return "{" + elements + "}"
        elif isinstance(expr, DictLiteral):
            if not expr.pairs:
                return "{}"
            pairs = ", ".join(
                f"{self._format_expression(k)}: {self._format_expression(v)}" for k, v in expr.pairs
            )
            return "{" + pairs + "}"
        elif isinstance(expr, TupleLiteral):
            elements = ", ".join(self._format_expression(e) for e in expr.elements)
            if len(expr.elements) == 1:
                return f"({elements},)"
            return f"({elements})"
        elif isinstance(expr, SomeExpression):
            return f"Some({self._format_expression(expr.value)})"
        elif isinstance(expr, OkExpression):
            return f"Ok({self._format_expression(expr.value)})"
        elif isinstance(expr, ErrExpression):
            return f"Err({self._format_expression(expr.value)})"
        elif isinstance(expr, UnwrapExpression):
            return f"{self._format_expression(expr.operand)}?"
        elif isinstance(expr, BinaryExpression):
            return self._format_binary_expression(expr)
        elif isinstance(expr, UnaryExpression):
            return self._format_unary_expression(expr)
        elif isinstance(expr, CallExpression):
            return self._format_call_expression(expr)
        elif isinstance(expr, MemberAccess):
            return f"{self._format_expression(expr.object)}.{expr.member}"
        elif isinstance(expr, IndexExpression):
            return f"{self._format_expression(expr.object)}[{self._format_expression(expr.index)}]"
        elif isinstance(expr, ConditionalExpression):
            cond = self._format_expression(expr.condition)
            then = self._format_expression(expr.then_expr)
            els = self._format_expression(expr.else_expr)
            return f"{then} if {cond} else {els}"
        elif isinstance(expr, LambdaExpression):
            return self._format_lambda_expression(expr)
        elif isinstance(expr, RangeExpression):
            return self._format_range_expression(expr)
        elif isinstance(expr, MatchExpression):
            return self._format_match_expression(expr)
        elif isinstance(expr, SelfExpression):
            return "self"
        elif isinstance(expr, EnumVariantAccess):
            return f"{expr.enum_name}::{expr.variant_name}"
        elif isinstance(expr, StructLiteral):
            return self._format_struct_literal(expr)
        else:
            return f"<unknown: {type(expr).__name__}>"

    def _format_binary_expression(self, expr: BinaryExpression) -> str:
        """Format a binary expression."""
        left = self._format_expression(expr.left)
        right = self._format_expression(expr.right)
        op = BINARY_OP_SYMBOLS.get(expr.operator, "?")

        if self.config.space_around_operators:
            return f"{left} {op} {right}"
        return f"{left}{op}{right}"

    def _format_unary_expression(self, expr: UnaryExpression) -> str:
        """Format a unary expression."""
        operand = self._format_expression(expr.operand)
        op = UNARY_OP_SYMBOLS.get(expr.operator, "?")
        return f"{op}{operand}"

    def _format_call_expression(self, expr: CallExpression) -> str:
        """Format a call expression."""
        callee = self._format_expression(expr.callee)
        args = ", ".join(self._format_expression(a) for a in expr.arguments)
        return f"{callee}({args})"

    def _format_lambda_expression(self, expr: LambdaExpression) -> str:
        """Format a lambda expression."""
        params = ", ".join(self._format_parameter(p) for p in expr.parameters)

        if isinstance(expr.body, Block):
            # Multi-line lambda - simplified representation
            return f"({params}) => {{ ... }}"
        else:
            body = self._format_expression(expr.body)
            return f"({params}) => {body}"

    def _format_range_expression(self, expr: RangeExpression) -> str:
        """Format a range expression."""
        start = self._format_expression(expr.start)
        end = self._format_expression(expr.end)
        op = "..=" if expr.inclusive else ".."
        return f"{start}{op}{end}"

    def _format_match_expression(self, expr: MatchExpression) -> str:
        """Format a match expression (simplified single-line for inline use)."""
        subject = self._format_expression(expr.subject)
        return f"match {subject} {{ ... }}"

    def _format_struct_literal(self, expr: StructLiteral) -> str:
        """Format a struct literal."""
        fields = ", ".join(
            f"{name}: {self._format_expression(value)}" for name, value in expr.fields
        )
        return f"{expr.struct_name} {{ {fields} }}"

    # -------------------------------------------------------------------------
    # Type and Parameter Formatting
    # -------------------------------------------------------------------------

    def _format_type_annotation(self, type_ann: TypeAnnotation) -> str:
        """Format a type annotation."""
        if isinstance(type_ann, SimpleType):
            return type_ann.name
        elif isinstance(type_ann, GenericType):
            args = ", ".join(self._format_type_annotation(a) for a in type_ann.type_args)
            return f"{type_ann.base}[{args}]"
        elif isinstance(type_ann, FunctionType):
            params = ", ".join(self._format_type_annotation(p) for p in type_ann.param_types)
            ret = self._format_type_annotation(type_ann.return_type)
            return f"({params}) -> {ret}"
        else:
            return str(type_ann)

    def _format_parameter(self, param: Parameter) -> str:
        """Format a function parameter."""
        parts = [param.name]

        if param.type_annotation:
            parts.append(f": {self._format_type_annotation(param.type_annotation)}")

        if param.default_value:
            parts.append(f" = {self._format_expression(param.default_value)}")

        return "".join(parts)

    def _format_pattern(self, pattern: Pattern) -> str:
        """Format a match pattern."""
        if isinstance(pattern, LiteralPattern):
            return self._format_expression(pattern.value)
        elif isinstance(pattern, IdentifierPattern):
            return "_" if pattern.is_wildcard else pattern.name
        elif isinstance(pattern, TuplePattern):
            elements = ", ".join(self._format_pattern(e) for e in pattern.elements)
            return f"({elements})"
        elif isinstance(pattern, ConstructorPattern):
            args = ", ".join(self._format_pattern(a) for a in pattern.args)
            return f"{pattern.name}({args})"
        elif isinstance(pattern, EnumPattern):
            if pattern.bindings:
                bindings = ", ".join(self._format_pattern(b) for b in pattern.bindings)
                return f"{pattern.enum_name}::{pattern.variant_name}({bindings})"
            return f"{pattern.enum_name}::{pattern.variant_name}"
        else:
            return "<unknown pattern>"


# =============================================================================
# Public API
# =============================================================================


def format_source(source: str, config: FormatConfig | None = None) -> str:
    """
    Format MathViz source code.

    Args:
        source: The MathViz source code to format
        config: Optional formatting configuration

    Returns:
        The formatted source code

    Raises:
        MathVizError: If the source cannot be parsed
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    formatter = Formatter(config)
    return formatter.format_program(ast)


def format_file(
    filepath: str | Path,
    config: FormatConfig | None = None,
    in_place: bool = False,
) -> str:
    """
    Format a MathViz file.

    Args:
        filepath: Path to the .mviz file
        config: Optional formatting configuration
        in_place: If True, write the formatted output back to the file

    Returns:
        The formatted source code

    Raises:
        FileNotFoundError: If the file does not exist
        MathVizError: If the source cannot be parsed
    """
    path = Path(filepath)
    source = path.read_text(encoding="utf-8")

    formatted = format_source(source, config)

    if in_place:
        path.write_text(formatted, encoding="utf-8")

    return formatted


def check_format(
    source: str,
    config: FormatConfig | None = None,
) -> bool:
    """
    Check if source code is properly formatted.

    Args:
        source: The MathViz source code to check
        config: Optional formatting configuration

    Returns:
        True if the source is properly formatted, False otherwise
    """
    try:
        formatted = format_source(source, config)
        return source == formatted
    except Exception:
        return False


def get_diff(
    source: str,
    config: FormatConfig | None = None,
    filename: str = "<input>",
) -> str:
    """
    Get a diff showing formatting changes.

    Args:
        source: The MathViz source code
        config: Optional formatting configuration
        filename: Filename for diff header

    Returns:
        A unified diff string, or empty string if no changes needed
    """
    try:
        formatted = format_source(source, config)

        if source == formatted:
            return ""

        diff = difflib.unified_diff(
            source.splitlines(keepends=True),
            formatted.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )

        return "".join(diff)
    except Exception as e:
        return f"Error: {e}"
