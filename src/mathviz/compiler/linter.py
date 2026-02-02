"""
MathViz Linter - Code quality analysis and warnings.

This module implements a comprehensive static analysis system for MathViz code,
detecting various code quality issues including:

- Unused variables, functions, and parameters
- Unreachable code after return/break/continue
- Mathematical anti-patterns (float equality, division by zero)
- Style issues (naming conventions, variable shadowing)
- Performance problems (loop invariants, redundant computations)

The linter integrates with the existing AST infrastructure and provides
detailed diagnostic messages with suggestions for fixes.

Example:
    linter = Linter()
    violations = linter.lint(program)
    for v in violations:
        print(f"{v.location}: [{v.rule.code}] {v.message}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

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
    MatchExpression,
    MatchArm,
    Pattern,
    LiteralPattern,
    IdentifierPattern,
    TuplePattern,
    ConstructorPattern,
    # Statements
    Statement,
    ExpressionStatement,
    LetStatement,
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
)
from mathviz.utils.errors import SourceLocation


# =============================================================================
# Lint Rule Configuration
# =============================================================================


class LintLevel(Enum):
    """
    Severity level for lint rules.

    ALLOW: Rule is disabled, no diagnostic produced
    WARN: Rule produces a warning (compilation continues)
    DENY: Rule produces an error (compilation may fail in strict mode)
    """

    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


class LintCategory(Enum):
    """
    Categories of lint rules for organization and filtering.
    """

    UNUSED = "unused"              # Unused variables, functions, imports
    UNREACHABLE = "unreachable"    # Dead/unreachable code
    STYLE = "style"                # Naming conventions, formatting
    PERFORMANCE = "performance"    # Performance anti-patterns
    CORRECTNESS = "correctness"    # Likely bugs or errors
    MATH = "math"                  # Mathematical anti-patterns


@dataclass(frozen=True)
class LintRule:
    """
    Definition of a single lint rule.

    Attributes:
        code: Unique rule identifier (e.g., "W0001")
        name: Human-readable rule name (e.g., "unused-variable")
        category: The category this rule belongs to
        message: Template message for the violation (use {} for placeholders)
        level: Default severity level
        suggestion: Optional template for a fix suggestion
    """

    code: str
    name: str
    category: LintCategory
    message: str
    level: LintLevel = LintLevel.WARN
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.code} ({self.name})"


@dataclass
class LintViolation:
    """
    A detected lint violation in the source code.

    Attributes:
        rule: The lint rule that was violated
        location: Source location of the violation
        message: Formatted message describing the issue
        suggestion: Optional suggestion for fixing the issue
        related_locations: Additional related source locations (e.g., definition site)
    """

    rule: LintRule
    location: Optional[SourceLocation]
    message: str
    suggestion: Optional[str] = None
    related_locations: list[SourceLocation] = field(default_factory=list)

    def __str__(self) -> str:
        loc_str = f"{self.location}" if self.location else "<unknown>"
        return f"[{self.rule.code}] {loc_str}: {self.message}"

    def format_full(self) -> str:
        """Format the violation with suggestion and related locations."""
        lines = [str(self)]
        if self.suggestion:
            lines.append(f"  suggestion: {self.suggestion}")
        for related in self.related_locations:
            lines.append(f"  related: {related}")
        return "\n".join(lines)


# =============================================================================
# Lint Rules - Unused Code
# =============================================================================

UNUSED_VARIABLE = LintRule(
    code="W0001",
    name="unused-variable",
    category=LintCategory.UNUSED,
    message="variable '{}' is never used",
    suggestion="if this is intentional, prefix with underscore: _{}",
)

UNUSED_FUNCTION = LintRule(
    code="W0002",
    name="unused-function",
    category=LintCategory.UNUSED,
    message="function '{}' is never called",
    suggestion="remove the function or add a call to it",
)

UNUSED_PARAMETER = LintRule(
    code="W0003",
    name="unused-parameter",
    category=LintCategory.UNUSED,
    message="parameter '{}' is never used in function '{}'",
    suggestion="if this is intentional, prefix with underscore: _{}",
)

UNUSED_IMPORT = LintRule(
    code="W0004",
    name="unused-import",
    category=LintCategory.UNUSED,
    message="import '{}' is never used",
    suggestion="remove the unused import",
)

UNUSED_LOOP_VARIABLE = LintRule(
    code="W0005",
    name="unused-loop-variable",
    category=LintCategory.UNUSED,
    message="loop variable '{}' is never used",
    suggestion="if this is intentional, use '_' as the variable name",
)


# =============================================================================
# Lint Rules - Unreachable Code
# =============================================================================

UNREACHABLE_CODE = LintRule(
    code="W0010",
    name="unreachable-code",
    category=LintCategory.UNREACHABLE,
    message="code after this statement is unreachable",
    level=LintLevel.WARN,
    suggestion="remove the unreachable code",
)

UNREACHABLE_PATTERN = LintRule(
    code="W0011",
    name="unreachable-pattern",
    category=LintCategory.UNREACHABLE,
    message="this pattern is unreachable (previous pattern catches all)",
    suggestion="remove this unreachable pattern or reorder match arms",
)

NON_EXHAUSTIVE_MATCH = LintRule(
    code="W0015",
    name="non-exhaustive-match",
    category=LintCategory.CORRECTNESS,
    message="non-exhaustive match: not all patterns are covered",
    level=LintLevel.WARN,
    suggestion="add a wildcard pattern `_` to handle remaining cases",
)

REDUNDANT_PATTERN = LintRule(
    code="W0016",
    name="redundant-pattern",
    category=LintCategory.UNREACHABLE,
    message="redundant pattern (already covered by previous patterns)",
    suggestion="remove this redundant pattern",
)

REDUNDANT_ASSIGNMENT = LintRule(
    code="W0017",
    name="redundant-assignment",
    category=LintCategory.CORRECTNESS,
    message="variable '{}' is assigned but never read before next assignment",
    suggestion="remove the redundant assignment or use the value",
)

ALWAYS_TRUE_CONDITION = LintRule(
    code="W0012",
    name="always-true-condition",
    category=LintCategory.UNREACHABLE,
    message="condition is always true",
    suggestion="simplify the control flow",
)

ALWAYS_FALSE_CONDITION = LintRule(
    code="W0013",
    name="always-false-condition",
    category=LintCategory.UNREACHABLE,
    message="condition is always false, code will never execute",
    suggestion="remove the unreachable branch",
)

EMPTY_BLOCK = LintRule(
    code="W0014",
    name="empty-block",
    category=LintCategory.UNREACHABLE,
    message="empty {} block",
    suggestion="add a pass statement or remove the empty block",
)


# =============================================================================
# Lint Rules - Mathematical Anti-patterns
# =============================================================================

FLOAT_EQUALITY = LintRule(
    code="W0020",
    name="float-equality",
    category=LintCategory.MATH,
    message="comparing floating-point numbers with == is unreliable",
    level=LintLevel.WARN,
    suggestion="use abs(a - b) < epsilon instead",
)

DIVISION_BY_ZERO_POSSIBLE = LintRule(
    code="W0021",
    name="possible-division-by-zero",
    category=LintCategory.MATH,
    message="possible division by zero",
    level=LintLevel.WARN,
    suggestion="check if the divisor could be zero",
)

INEFFICIENT_POWER = LintRule(
    code="W0022",
    name="inefficient-power",
    category=LintCategory.PERFORMANCE,
    message="x^{} can be more efficiently written as {}",
    suggestion="use multiplication for small integer exponents",
)

UNNECESSARY_MATH_CALL = LintRule(
    code="W0023",
    name="unnecessary-math-call",
    category=LintCategory.PERFORMANCE,
    message="unnecessary call to math function on constant",
    suggestion="precompute the constant value",
)

INTEGER_DIVISION_TRUNCATION = LintRule(
    code="W0024",
    name="integer-division-truncation",
    category=LintCategory.MATH,
    message="division of integers may silently truncate",
    suggestion="use float division (/) or explicit floor division (//)",
)


# =============================================================================
# Lint Rules - Style
# =============================================================================

NAMING_CONVENTION = LintRule(
    code="W0030",
    name="naming-convention",
    category=LintCategory.STYLE,
    message="'{}' should use {} naming convention",
)

SHADOWING = LintRule(
    code="W0031",
    name="variable-shadowing",
    category=LintCategory.STYLE,
    message="variable '{}' shadows a variable from outer scope",
    level=LintLevel.WARN,
    suggestion="use a different variable name to avoid confusion",
)

MUTABLE_DEFAULT_ARGUMENT = LintRule(
    code="W0032",
    name="mutable-default-argument",
    category=LintCategory.CORRECTNESS,
    message="mutable default argument '{}' may cause unexpected behavior",
    level=LintLevel.WARN,
    suggestion="use None as default and create the mutable inside the function",
)

COMPARISON_TO_NONE = LintRule(
    code="W0033",
    name="comparison-to-none",
    category=LintCategory.STYLE,
    message="comparison to None should use 'is' or 'is not'",
    suggestion="use 'x is None' instead of 'x == None'",
)

REDUNDANT_BOOLEAN = LintRule(
    code="W0034",
    name="redundant-boolean",
    category=LintCategory.STYLE,
    message="redundant boolean comparison",
    suggestion="use the boolean value directly",
)


# =============================================================================
# Lint Rules - Performance
# =============================================================================

LOOP_INVARIANT = LintRule(
    code="W0040",
    name="loop-invariant",
    category=LintCategory.PERFORMANCE,
    message="expression does not depend on loop variable, consider moving outside loop",
)

REDUNDANT_COMPUTATION = LintRule(
    code="W0041",
    name="redundant-computation",
    category=LintCategory.PERFORMANCE,
    message="this computation is performed multiple times with same inputs",
    suggestion="store the result in a variable",
)

INEFFICIENT_LIST_APPEND = LintRule(
    code="W0042",
    name="inefficient-list-construction",
    category=LintCategory.PERFORMANCE,
    message="building list with append in loop is less efficient than list comprehension",
    suggestion="consider using a list comprehension",
)

STRING_CONCAT_IN_LOOP = LintRule(
    code="W0043",
    name="string-concat-in-loop",
    category=LintCategory.PERFORMANCE,
    message="string concatenation in loop is inefficient",
    suggestion="use ''.join() or a list to accumulate strings",
)


# =============================================================================
# Lint Rules - Correctness
# =============================================================================

SELF_ASSIGNMENT = LintRule(
    code="W0050",
    name="self-assignment",
    category=LintCategory.CORRECTNESS,
    message="variable '{}' is assigned to itself",
    level=LintLevel.WARN,
)

COMPARISON_WITH_ITSELF = LintRule(
    code="W0051",
    name="comparison-with-itself",
    category=LintCategory.CORRECTNESS,
    message="comparing '{}' with itself",
    level=LintLevel.WARN,
)

UNREACHABLE_RETURN = LintRule(
    code="W0052",
    name="unreachable-return",
    category=LintCategory.CORRECTNESS,
    message="function may not return a value on all paths",
    level=LintLevel.WARN,
)

MISSING_RETURN = LintRule(
    code="W0053",
    name="missing-return",
    category=LintCategory.CORRECTNESS,
    message="function '{}' with return type may not return a value",
)


# =============================================================================
# Rule Registry
# =============================================================================


ALL_RULES: dict[str, LintRule] = {
    # Unused
    UNUSED_VARIABLE.code: UNUSED_VARIABLE,
    UNUSED_FUNCTION.code: UNUSED_FUNCTION,
    UNUSED_PARAMETER.code: UNUSED_PARAMETER,
    UNUSED_IMPORT.code: UNUSED_IMPORT,
    UNUSED_LOOP_VARIABLE.code: UNUSED_LOOP_VARIABLE,
    # Unreachable
    UNREACHABLE_CODE.code: UNREACHABLE_CODE,
    UNREACHABLE_PATTERN.code: UNREACHABLE_PATTERN,
    NON_EXHAUSTIVE_MATCH.code: NON_EXHAUSTIVE_MATCH,
    REDUNDANT_PATTERN.code: REDUNDANT_PATTERN,
    REDUNDANT_ASSIGNMENT.code: REDUNDANT_ASSIGNMENT,
    ALWAYS_TRUE_CONDITION.code: ALWAYS_TRUE_CONDITION,
    ALWAYS_FALSE_CONDITION.code: ALWAYS_FALSE_CONDITION,
    EMPTY_BLOCK.code: EMPTY_BLOCK,
    # Math
    FLOAT_EQUALITY.code: FLOAT_EQUALITY,
    DIVISION_BY_ZERO_POSSIBLE.code: DIVISION_BY_ZERO_POSSIBLE,
    INEFFICIENT_POWER.code: INEFFICIENT_POWER,
    UNNECESSARY_MATH_CALL.code: UNNECESSARY_MATH_CALL,
    INTEGER_DIVISION_TRUNCATION.code: INTEGER_DIVISION_TRUNCATION,
    # Style
    NAMING_CONVENTION.code: NAMING_CONVENTION,
    SHADOWING.code: SHADOWING,
    MUTABLE_DEFAULT_ARGUMENT.code: MUTABLE_DEFAULT_ARGUMENT,
    COMPARISON_TO_NONE.code: COMPARISON_TO_NONE,
    REDUNDANT_BOOLEAN.code: REDUNDANT_BOOLEAN,
    # Performance
    LOOP_INVARIANT.code: LOOP_INVARIANT,
    REDUNDANT_COMPUTATION.code: REDUNDANT_COMPUTATION,
    INEFFICIENT_LIST_APPEND.code: INEFFICIENT_LIST_APPEND,
    STRING_CONCAT_IN_LOOP.code: STRING_CONCAT_IN_LOOP,
    # Correctness
    SELF_ASSIGNMENT.code: SELF_ASSIGNMENT,
    COMPARISON_WITH_ITSELF.code: COMPARISON_WITH_ITSELF,
    UNREACHABLE_RETURN.code: UNREACHABLE_RETURN,
    MISSING_RETURN.code: MISSING_RETURN,
}

# Also index by name
RULES_BY_NAME: dict[str, LintRule] = {
    rule.name: rule for rule in ALL_RULES.values()
}


# =============================================================================
# Lint Configuration
# =============================================================================


@dataclass
class LintConfiguration:
    """
    Configuration for the linter specifying rule levels.

    Allows customizing which rules are enabled/disabled and their severity.

    Example:
        config = LintConfiguration()
        config.set_level("unused-variable", LintLevel.ALLOW)
        config.set_level_by_category(LintCategory.STYLE, LintLevel.DENY)
    """

    rule_levels: dict[str, LintLevel] = field(default_factory=dict)

    def get_level(self, rule: LintRule) -> LintLevel:
        """Get the effective level for a rule."""
        # Check by code first, then by name
        if rule.code in self.rule_levels:
            return self.rule_levels[rule.code]
        if rule.name in self.rule_levels:
            return self.rule_levels[rule.name]
        return rule.level  # Default level

    def set_level(self, rule_id: str, level: LintLevel) -> None:
        """Set the level for a rule by code or name."""
        self.rule_levels[rule_id] = level

    def set_level_by_category(self, category: LintCategory, level: LintLevel) -> None:
        """Set the level for all rules in a category."""
        for rule in ALL_RULES.values():
            if rule.category == category:
                self.rule_levels[rule.code] = level

    def allow(self, rule_id: str) -> None:
        """Disable a rule."""
        self.set_level(rule_id, LintLevel.ALLOW)

    def warn(self, rule_id: str) -> None:
        """Set a rule to warning level."""
        self.set_level(rule_id, LintLevel.WARN)

    def deny(self, rule_id: str) -> None:
        """Set a rule to error level."""
        self.set_level(rule_id, LintLevel.DENY)

    def allow_all(self) -> None:
        """Disable all rules."""
        for rule in ALL_RULES.values():
            self.rule_levels[rule.code] = LintLevel.ALLOW

    def warn_all(self) -> None:
        """Set all rules to warning level."""
        for rule in ALL_RULES.values():
            self.rule_levels[rule.code] = LintLevel.WARN

    def deny_all(self) -> None:
        """Set all rules to error level."""
        for rule in ALL_RULES.values():
            self.rule_levels[rule.code] = LintLevel.DENY

    @classmethod
    def parse_directive(cls, directive: str) -> tuple[str, str, LintLevel]:
        """
        Parse a lint directive from source code.

        Formats:
            #![allow(rule-name)]
            #![warn(rule-name)]
            #![deny(rule-name)]

        Returns:
            Tuple of (action, rule_name, level)

        Raises:
            ValueError: If directive format is invalid
        """
        pattern = r"#!\[(allow|warn|deny)\(([a-zA-Z0-9_-]+)\)\]"
        match = re.match(pattern, directive.strip())
        if not match:
            raise ValueError(f"Invalid lint directive: {directive}")

        action = match.group(1)
        rule_name = match.group(2)
        level_map = {
            "allow": LintLevel.ALLOW,
            "warn": LintLevel.WARN,
            "deny": LintLevel.DENY,
        }
        return action, rule_name, level_map[action]


# =============================================================================
# Variable/Symbol Tracking
# =============================================================================


@dataclass
class SymbolInfo:
    """Information about a defined symbol."""

    name: str
    location: Optional[SourceLocation]
    kind: str  # "variable", "function", "parameter", "import", "loop_variable"
    used: bool = False
    scope_depth: int = 0


@dataclass
class ScopeInfo:
    """Information about a scope level."""

    name: str
    depth: int
    symbols: dict[str, SymbolInfo] = field(default_factory=dict)
    parent: Optional["ScopeInfo"] = None


# =============================================================================
# Main Linter Implementation
# =============================================================================


class Linter(BaseASTVisitor):
    """
    Main linter class that runs all lint checks on a MathViz program.

    The linter performs multi-pass analysis:
    1. Collect definitions (variables, functions, imports)
    2. Collect usages (references to defined symbols)
    3. Check unused code
    4. Check unreachable code
    5. Check mathematical issues
    6. Check style issues
    7. Check performance issues

    Example:
        linter = Linter()
        violations = linter.lint(program)
        for v in violations:
            print(v)
    """

    def __init__(self, config: Optional[LintConfiguration] = None) -> None:
        """
        Initialize the linter.

        Args:
            config: Optional lint configuration for customizing rule levels
        """
        self.config = config or LintConfiguration()
        self.violations: list[LintViolation] = []

        # Symbol tracking
        self._scopes: list[ScopeInfo] = []
        self._current_scope: Optional[ScopeInfo] = None
        self._global_symbols: dict[str, SymbolInfo] = {}

        # Function tracking
        self._defined_functions: dict[str, SymbolInfo] = {}
        self._called_functions: set[str] = set()
        self._current_function: Optional[str] = None
        self._function_params: dict[str, dict[str, SymbolInfo]] = {}

        # Import tracking
        self._imports: dict[str, SymbolInfo] = {}
        self._used_imports: set[str] = set()

        # Control flow tracking
        self._after_return = False
        self._after_break = False
        self._after_continue = False
        self._in_loop = False
        self._loop_variables: set[str] = set()

        # Expression analysis state
        self._in_condition = False
        self._analyzing_loop_body = False
        self._current_loop_var: Optional[str] = None

    def lint(self, program: Program) -> list[LintViolation]:
        """
        Run all lint checks on a program.

        Args:
            program: The parsed AST to analyze

        Returns:
            List of lint violations found
        """
        self.violations = []
        self._reset_state()

        # Pass 1: Collect all definitions
        self._collect_definitions(program)

        # Pass 2: Collect all usages and check for issues
        self._analyze_program(program)

        # Pass 3: Check for unused definitions
        self._check_unused()

        return self.violations

    def _reset_state(self) -> None:
        """Reset all internal state for a new lint run."""
        self._scopes = []
        self._current_scope = None
        self._global_symbols = {}
        self._defined_functions = {}
        self._called_functions = set()
        self._current_function = None
        self._function_params = {}
        self._imports = {}
        self._used_imports = set()
        self._after_return = False
        self._after_break = False
        self._after_continue = False
        self._in_loop = False
        self._loop_variables = set()

    def _emit(
        self,
        rule: LintRule,
        location: Optional[SourceLocation],
        *format_args: str,
        suggestion: Optional[str] = None,
        related_locations: Optional[list[SourceLocation]] = None,
    ) -> None:
        """
        Emit a lint violation if the rule is enabled.

        Args:
            rule: The lint rule being violated
            location: Source location of the violation
            format_args: Arguments to format into the rule message
            suggestion: Optional fix suggestion
            related_locations: Additional related source locations
        """
        level = self.config.get_level(rule)
        if level == LintLevel.ALLOW:
            return

        message = rule.message.format(*format_args) if format_args else rule.message
        suggestion_text = suggestion
        if suggestion_text is None and rule.suggestion:
            if format_args:
                try:
                    suggestion_text = rule.suggestion.format(*format_args)
                except (IndexError, KeyError):
                    suggestion_text = rule.suggestion
            else:
                suggestion_text = rule.suggestion

        self.violations.append(LintViolation(
            rule=rule,
            location=location,
            message=message,
            suggestion=suggestion_text,
            related_locations=related_locations or [],
        ))

    # =========================================================================
    # Scope Management
    # =========================================================================

    def _enter_scope(self, name: str) -> None:
        """Enter a new scope level."""
        depth = len(self._scopes)
        scope = ScopeInfo(name=name, depth=depth, parent=self._current_scope)
        self._scopes.append(scope)
        self._current_scope = scope

    def _exit_scope(self) -> None:
        """Exit the current scope level."""
        if self._scopes:
            self._scopes.pop()
            self._current_scope = self._scopes[-1] if self._scopes else None

    def _define_symbol(
        self,
        name: str,
        location: Optional[SourceLocation],
        kind: str,
    ) -> None:
        """Define a symbol in the current scope."""
        if self._current_scope is None:
            # Global scope
            self._global_symbols[name] = SymbolInfo(
                name=name,
                location=location,
                kind=kind,
                scope_depth=0,
            )
        else:
            # Check for shadowing
            if not name.startswith("_"):
                outer_symbol = self._lookup_symbol(name)
                if outer_symbol is not None:
                    self._emit(
                        SHADOWING,
                        location,
                        name,
                        related_locations=[outer_symbol.location]
                        if outer_symbol.location else [],
                    )

            self._current_scope.symbols[name] = SymbolInfo(
                name=name,
                location=location,
                kind=kind,
                scope_depth=self._current_scope.depth,
            )

    def _lookup_symbol(self, name: str) -> Optional[SymbolInfo]:
        """Look up a symbol in the scope chain."""
        # Check current scope chain
        scope = self._current_scope
        while scope:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent

        # Check global scope
        if name in self._global_symbols:
            return self._global_symbols[name]

        return None

    def _mark_symbol_used(self, name: str) -> None:
        """Mark a symbol as used."""
        # Check local scopes first
        scope = self._current_scope
        while scope:
            if name in scope.symbols:
                scope.symbols[name].used = True
                # Also sync with function params if this is a parameter
                if (self._current_function and
                    self._current_function in self._function_params and
                    name in self._function_params[self._current_function]):
                    self._function_params[self._current_function][name].used = True
                return
            scope = scope.parent

        # Check global symbols
        if name in self._global_symbols:
            self._global_symbols[name].used = True
            return

        # Check function parameters (fallback)
        if self._current_function and self._current_function in self._function_params:
            params = self._function_params[self._current_function]
            if name in params:
                params[name].used = True
                return

        # Check imports
        if name in self._imports:
            self._imports[name].used = True
            self._used_imports.add(name)

    # =========================================================================
    # Pass 1: Collect Definitions
    # =========================================================================

    def _collect_definitions(self, program: Program) -> None:
        """Collect all definitions in the program."""
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                self._defined_functions[stmt.name] = SymbolInfo(
                    name=stmt.name,
                    location=stmt.location,
                    kind="function",
                )
                # Collect parameter definitions
                self._function_params[stmt.name] = {}
                for param in stmt.parameters:
                    self._function_params[stmt.name][param.name] = SymbolInfo(
                        name=param.name,
                        location=param.location,
                        kind="parameter",
                    )
            elif isinstance(stmt, ImportStatement):
                self._collect_import(stmt)
            elif isinstance(stmt, UseStatement):
                self._collect_use(stmt)
            elif isinstance(stmt, ClassDef):
                self._defined_functions[stmt.name] = SymbolInfo(
                    name=stmt.name,
                    location=stmt.location,
                    kind="class",
                )
            elif isinstance(stmt, SceneDef):
                self._defined_functions[stmt.name] = SymbolInfo(
                    name=stmt.name,
                    location=stmt.location,
                    kind="scene",
                )

    def _collect_import(self, stmt: ImportStatement) -> None:
        """Collect import definitions."""
        if stmt.is_from_import:
            for name, alias in stmt.names:
                import_name = alias or name
                self._imports[import_name] = SymbolInfo(
                    name=import_name,
                    location=stmt.location,
                    kind="import",
                )
        else:
            import_name = stmt.alias or stmt.module
            self._imports[import_name] = SymbolInfo(
                name=import_name,
                location=stmt.location,
                kind="import",
            )

    def _collect_use(self, stmt: UseStatement) -> None:
        """Collect use statement definitions."""
        if stmt.alias:
            self._imports[stmt.alias] = SymbolInfo(
                name=stmt.alias,
                location=stmt.location,
                kind="import",
            )
        elif not stmt.wildcard and stmt.module_path:
            name = stmt.module_path[-1]
            self._imports[name] = SymbolInfo(
                name=name,
                location=stmt.location,
                kind="import",
            )

    # =========================================================================
    # Pass 2: Analyze Program
    # =========================================================================

    def _analyze_program(self, program: Program) -> None:
        """Analyze the program for lint issues."""
        self.visit(program)

    def visit_program(self, node: Program) -> None:
        """Visit the program root."""
        for stmt in node.statements:
            self.visit(stmt)

    def visit_block(self, node: Block) -> None:
        """Visit a block of statements, checking for unreachable code."""
        self._after_return = False
        self._after_break = False
        self._after_continue = False

        # Check for empty block
        if not node.statements:
            self._emit(EMPTY_BLOCK, node.location, "statement")

        for i, stmt in enumerate(node.statements):
            # Check if previous statement made this unreachable
            if self._after_return or self._after_break or self._after_continue:
                self._emit(UNREACHABLE_CODE, stmt.location)
                break  # Don't report multiple unreachable warnings

            self.visit(stmt)

            # Update control flow state
            if isinstance(stmt, ReturnStatement):
                self._after_return = True
            elif isinstance(stmt, BreakStatement):
                self._after_break = True
            elif isinstance(stmt, ContinueStatement):
                self._after_continue = True

    def visit_function_def(self, node: FunctionDef) -> None:
        """Visit a function definition."""
        old_function = self._current_function
        self._current_function = node.name

        # Enter function scope
        self._enter_scope(f"function:{node.name}")

        # Define parameters in scope
        for param in node.parameters:
            self._define_symbol(param.name, param.location, "parameter")

            # Check parameter naming convention
            self._check_naming_convention(param.name, "parameter", param.location)

            # Check for mutable default arguments
            if param.default_value:
                if isinstance(param.default_value, (ListLiteral, SetLiteral, DictLiteral)):
                    self._emit(
                        MUTABLE_DEFAULT_ARGUMENT,
                        param.location,
                        param.name,
                    )
                # Visit default value to track usages
                self.visit(param.default_value)

        # Check function naming convention
        self._check_naming_convention(node.name, "function", node.location)

        # Visit function body
        if node.body:
            self._after_return = False
            self.visit(node.body)

            # Check if function with return type might not return
            if node.return_type and not self._after_return:
                # Check if it's not a void/None return type
                if isinstance(node.return_type, SimpleType):
                    if node.return_type.name not in ("None", "Void"):
                        self._emit(MISSING_RETURN, node.location, node.name)

        self._exit_scope()
        self._current_function = old_function

        # Check unused parameters
        if node.name in self._function_params:
            for param_name, param_info in self._function_params[node.name].items():
                if not param_info.used and not param_name.startswith("_"):
                    # Don't warn about 'self' parameter
                    if param_name != "self":
                        self._emit(
                            UNUSED_PARAMETER,
                            param_info.location,
                            param_name,
                            node.name,
                        )

    def visit_let_statement(self, node: LetStatement) -> None:
        """Visit a variable declaration."""
        # Visit the value first (before defining the variable)
        if node.value:
            self.visit(node.value)

            # Check for self-assignment
            if isinstance(node.value, Identifier) and node.value.name == node.name:
                self._emit(SELF_ASSIGNMENT, node.location, node.name)

        # Define the variable
        self._define_symbol(node.name, node.location, "variable")

        # Check naming convention
        self._check_naming_convention(node.name, "variable", node.location)

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        """Visit an assignment statement."""
        # Visit value first
        self.visit(node.value)

        # Check for self-assignment
        if isinstance(node.target, Identifier) and isinstance(node.value, Identifier):
            if node.target.name == node.value.name:
                self._emit(SELF_ASSIGNMENT, node.location, node.target.name)

        # Visit target
        self.visit(node.target)

    def visit_for_statement(self, node: ForStatement) -> None:
        """Visit a for loop."""
        old_in_loop = self._in_loop
        old_loop_var = self._current_loop_var
        self._in_loop = True
        self._current_loop_var = node.variable

        # Visit iterable
        self.visit(node.iterable)

        # Enter loop scope
        self._enter_scope("for")

        # Define loop variable
        self._define_symbol(node.variable, node.location, "loop_variable")
        self._loop_variables.add(node.variable)

        # Visit loop body
        self._analyzing_loop_body = True
        old_after_break = self._after_break
        old_after_continue = self._after_continue
        self._after_break = False
        self._after_continue = False

        self.visit(node.body)

        self._after_break = old_after_break
        self._after_continue = old_after_continue
        self._analyzing_loop_body = False

        # Check if loop variable was used
        if self._current_scope and node.variable in self._current_scope.symbols:
            symbol = self._current_scope.symbols[node.variable]
            if not symbol.used and not node.variable.startswith("_") and node.variable != "_":
                self._emit(UNUSED_LOOP_VARIABLE, node.location, node.variable)

        self._exit_scope()
        self._loop_variables.discard(node.variable)
        self._current_loop_var = old_loop_var
        self._in_loop = old_in_loop

    def visit_while_statement(self, node: WhileStatement) -> None:
        """Visit a while loop."""
        old_in_loop = self._in_loop
        self._in_loop = True

        # Check condition
        self._in_condition = True
        self.visit(node.condition)
        self._in_condition = False

        # Check for always-true or always-false conditions
        self._check_constant_condition(node.condition)

        # Enter loop scope
        self._enter_scope("while")

        # Visit loop body
        old_after_break = self._after_break
        old_after_continue = self._after_continue
        self._after_break = False
        self._after_continue = False

        self.visit(node.body)

        self._after_break = old_after_break
        self._after_continue = old_after_continue

        self._exit_scope()
        self._in_loop = old_in_loop

    def visit_if_statement(self, node: IfStatement) -> None:
        """Visit an if statement."""
        # Check condition
        self._in_condition = True
        self.visit(node.condition)
        self._in_condition = False

        # Check for always-true or always-false conditions
        self._check_constant_condition(node.condition)

        # Visit then block
        self._enter_scope("if:then")
        self.visit(node.then_block)
        self._exit_scope()

        # Visit elif clauses
        for elif_cond, elif_block in node.elif_clauses:
            self._in_condition = True
            self.visit(elif_cond)
            self._in_condition = False

            self._check_constant_condition(elif_cond)

            self._enter_scope("if:elif")
            self.visit(elif_block)
            self._exit_scope()

        # Visit else block
        if node.else_block:
            self._enter_scope("if:else")
            self.visit(node.else_block)
            self._exit_scope()

    def visit_identifier(self, node: Identifier) -> None:
        """Visit an identifier (variable reference)."""
        self._mark_symbol_used(node.name)

    def visit_call_expression(self, node: CallExpression) -> None:
        """Visit a function call."""
        # Track called functions
        if isinstance(node.callee, Identifier):
            func_name = node.callee.name
            self._called_functions.add(func_name)
            self._mark_symbol_used(func_name)

            # Check if it's an import
            if func_name in self._imports:
                self._used_imports.add(func_name)

        # Visit callee and arguments
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        """Visit a binary expression, checking for mathematical issues."""
        self.visit(node.left)
        self.visit(node.right)

        # Check for float equality
        if node.operator in (BinaryOperator.EQ, BinaryOperator.NE):
            if self._is_float_expression(node.left) or self._is_float_expression(node.right):
                self._emit(FLOAT_EQUALITY, node.location)

            # Check for comparison to None with ==
            if self._is_none_expression(node.left) or self._is_none_expression(node.right):
                self._emit(COMPARISON_TO_NONE, node.location)

            # Check for comparison with itself
            if self._expressions_equal(node.left, node.right):
                if isinstance(node.left, Identifier):
                    self._emit(COMPARISON_WITH_ITSELF, node.location, node.left.name)

        # Check for division by zero
        if node.operator in (BinaryOperator.DIV, BinaryOperator.FLOOR_DIV, BinaryOperator.MOD):
            if self._is_zero_expression(node.right):
                self._emit(DIVISION_BY_ZERO_POSSIBLE, node.location)

        # Check for inefficient power
        if node.operator == BinaryOperator.POW:
            if isinstance(node.right, IntegerLiteral):
                exp = node.right.value
                if 2 <= exp <= 4:
                    suggestion = self._generate_power_suggestion(node.left, exp)
                    self._emit(
                        INEFFICIENT_POWER,
                        node.location,
                        str(exp),
                        suggestion,
                    )

        # Check for redundant boolean comparison
        if node.operator in (BinaryOperator.EQ, BinaryOperator.NE):
            if isinstance(node.right, BooleanLiteral) or isinstance(node.left, BooleanLiteral):
                self._emit(REDUNDANT_BOOLEAN, node.location)

    def visit_member_access(self, node: MemberAccess) -> None:
        """Visit a member access expression."""
        self.visit(node.object)

        # Track module usage
        if isinstance(node.object, Identifier):
            if node.object.name in self._imports:
                self._used_imports.add(node.object.name)

    def visit_import_statement(self, node: ImportStatement) -> None:
        """Visit an import statement."""
        # Already collected in pass 1
        pass

    def visit_use_statement(self, node: UseStatement) -> None:
        """Visit a use statement."""
        # Already collected in pass 1
        pass

    def visit_class_def(self, node: ClassDef) -> None:
        """Visit a class definition."""
        self._enter_scope(f"class:{node.name}")

        # Define 'self' in class scope
        self._define_symbol("self", node.location, "variable")

        if node.body:
            self.visit(node.body)

        self._exit_scope()

    def visit_scene_def(self, node: SceneDef) -> None:
        """Visit a scene definition."""
        # Mark scene as "used" since it's a top-level definition
        if node.name in self._defined_functions:
            self._defined_functions[node.name].used = True

        self._enter_scope(f"scene:{node.name}")

        # Define 'self' in scene scope
        self._define_symbol("self", node.location, "variable")

        if node.body:
            self.visit(node.body)

        self._exit_scope()

    def visit_module_decl(self, node: ModuleDecl) -> None:
        """Visit a module declaration."""
        self._enter_scope(f"module:{node.name}")
        self.visit(node.body)
        self._exit_scope()

    def visit_return_statement(self, node: ReturnStatement) -> None:
        """Visit a return statement."""
        if node.value:
            self.visit(node.value)
        self._after_return = True

    def visit_break_statement(self, node: BreakStatement) -> None:
        """Visit a break statement."""
        self._after_break = True

    def visit_continue_statement(self, node: ContinueStatement) -> None:
        """Visit a continue statement."""
        self._after_continue = True

    def visit_lambda_expression(self, node: LambdaExpression) -> None:
        """Visit a lambda expression."""
        self._enter_scope("lambda")

        # Define parameters
        for param in node.parameters:
            self._define_symbol(param.name, param.location, "parameter")

        # Visit body
        if isinstance(node.body, Block):
            self.visit(node.body)
        else:
            self.visit(node.body)

        self._exit_scope()

    def visit_match_expression(self, node: MatchExpression) -> None:
        """
        Visit a match expression, checking for:
        - Non-exhaustive patterns
        - Unreachable patterns (after a catch-all)
        - Redundant patterns
        """
        # Visit the subject expression
        self.visit(node.subject)

        # Track pattern coverage for exhaustiveness checking
        has_wildcard = False
        wildcard_location: Optional[SourceLocation] = None
        seen_literal_patterns: set[tuple[str, object]] = set()  # (type, value)

        for arm in node.arms:
            pattern = arm.pattern

            # Check if this pattern is after a catch-all wildcard
            if has_wildcard and wildcard_location:
                self._emit(
                    UNREACHABLE_PATTERN,
                    pattern.location,
                    related_locations=[wildcard_location],
                )
                # Don't analyze further patterns after catch-all
                continue

            # Check pattern type
            if isinstance(pattern, IdentifierPattern):
                if pattern.is_wildcard or pattern.name == "_":
                    # This is a catch-all pattern
                    has_wildcard = True
                    wildcard_location = pattern.location
                else:
                    # Regular identifier pattern - also catches all
                    has_wildcard = True
                    wildcard_location = pattern.location
                    # Define the bound variable
                    self._enter_scope("match_arm")
                    self._define_symbol(pattern.name, pattern.location, "variable")

            elif isinstance(pattern, LiteralPattern):
                # Check for duplicate literal patterns
                literal_key = self._get_literal_key(pattern.value)
                if literal_key in seen_literal_patterns:
                    self._emit(REDUNDANT_PATTERN, pattern.location)
                else:
                    seen_literal_patterns.add(literal_key)

            elif isinstance(pattern, TuplePattern):
                # Enter scope for bound names in tuple pattern
                self._enter_scope("match_arm")
                self._define_pattern_bindings(pattern)

            elif isinstance(pattern, ConstructorPattern):
                # Enter scope for bound names in constructor pattern
                self._enter_scope("match_arm")
                self._define_pattern_bindings(pattern)

            # Visit the guard if present
            if arm.guard:
                self.visit(arm.guard)

            # Visit the arm body
            if isinstance(arm.body, Block):
                self.visit(arm.body)
            else:
                self.visit(arm.body)

            # Exit scope if we entered one
            if isinstance(pattern, (TuplePattern, ConstructorPattern)):
                self._exit_scope()
            elif isinstance(pattern, IdentifierPattern) and not pattern.is_wildcard:
                self._exit_scope()

        # Check for non-exhaustive match (no wildcard pattern)
        if not has_wildcard and len(node.arms) > 0:
            # Only warn if there are arms but no catch-all
            # This is a heuristic - full exhaustiveness checking requires type info
            self._emit(NON_EXHAUSTIVE_MATCH, node.location)

    def _get_literal_key(self, expr: Expression) -> tuple[str, object]:
        """Get a hashable key for a literal expression."""
        if isinstance(expr, IntegerLiteral):
            return ("int", expr.value)
        elif isinstance(expr, FloatLiteral):
            return ("float", expr.value)
        elif isinstance(expr, StringLiteral):
            return ("str", expr.value)
        elif isinstance(expr, BooleanLiteral):
            return ("bool", expr.value)
        elif isinstance(expr, NoneLiteral):
            return ("none", None)
        else:
            return ("unknown", id(expr))

    def _define_pattern_bindings(self, pattern: Pattern) -> None:
        """Define variables bound by a pattern in the current scope."""
        if isinstance(pattern, IdentifierPattern):
            if not pattern.is_wildcard and pattern.name != "_":
                self._define_symbol(pattern.name, pattern.location, "variable")
        elif isinstance(pattern, TuplePattern):
            for elem in pattern.elements:
                self._define_pattern_bindings(elem)
        elif isinstance(pattern, ConstructorPattern):
            for arg in pattern.args:
                self._define_pattern_bindings(arg)
        # LiteralPattern doesn't bind any variables

    # =========================================================================
    # Pass 3: Check Unused
    # =========================================================================

    def _check_unused(self) -> None:
        """Check for unused definitions."""
        # Check unused functions
        for func_name, func_info in self._defined_functions.items():
            if not func_info.used and func_name not in self._called_functions:
                # Don't warn about 'main' function or private functions
                if func_name != "main" and not func_name.startswith("_"):
                    # Don't warn about classes and scenes
                    if func_info.kind == "function":
                        self._emit(UNUSED_FUNCTION, func_info.location, func_name)

        # Check unused imports
        for import_name, import_info in self._imports.items():
            if not import_info.used and import_name not in self._used_imports:
                if not import_name.startswith("_"):
                    self._emit(UNUSED_IMPORT, import_info.location, import_name)

        # Check unused global variables
        for var_name, var_info in self._global_symbols.items():
            if not var_info.used and var_info.kind == "variable":
                if not var_name.startswith("_"):
                    self._emit(UNUSED_VARIABLE, var_info.location, var_name)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _is_float_expression(self, expr: Expression) -> bool:
        """Check if an expression is or produces a float value."""
        if isinstance(expr, FloatLiteral):
            return True
        if isinstance(expr, BinaryExpression):
            if expr.operator == BinaryOperator.DIV:
                return True  # Division always produces float
            return (self._is_float_expression(expr.left) or
                    self._is_float_expression(expr.right))
        if isinstance(expr, CallExpression):
            if isinstance(expr.callee, Identifier):
                # Math functions typically return float
                float_funcs = {"sqrt", "sin", "cos", "tan", "exp", "log", "log10",
                               "pow", "abs", "float"}
                return expr.callee.name in float_funcs
        return False

    def _is_none_expression(self, expr: Expression) -> bool:
        """Check if an expression is None."""
        return isinstance(expr, NoneLiteral)

    def _is_zero_expression(self, expr: Expression) -> bool:
        """Check if an expression is definitely zero."""
        if isinstance(expr, IntegerLiteral):
            return expr.value == 0
        if isinstance(expr, FloatLiteral):
            return expr.value == 0.0
        return False

    def _expressions_equal(self, left: Expression, right: Expression) -> bool:
        """Check if two expressions are structurally equal."""
        if type(left) != type(right):
            return False
        if isinstance(left, Identifier) and isinstance(right, Identifier):
            return left.name == right.name
        if isinstance(left, IntegerLiteral) and isinstance(right, IntegerLiteral):
            return left.value == right.value
        if isinstance(left, FloatLiteral) and isinstance(right, FloatLiteral):
            return left.value == right.value
        if isinstance(left, StringLiteral) and isinstance(right, StringLiteral):
            return left.value == right.value
        return False

    def _generate_power_suggestion(self, base: Expression, exponent: int) -> str:
        """Generate a suggestion for replacing power with multiplication."""
        if isinstance(base, Identifier):
            base_str = base.name
        else:
            base_str = "x"

        if exponent == 2:
            return f"{base_str} * {base_str}"
        elif exponent == 3:
            return f"{base_str} * {base_str} * {base_str}"
        elif exponent == 4:
            return f"({base_str} * {base_str}) * ({base_str} * {base_str})"
        return f"{base_str}^{exponent}"

    def _check_constant_condition(self, expr: Expression) -> None:
        """Check if a condition is always true or false."""
        if isinstance(expr, BooleanLiteral):
            if expr.value:
                self._emit(ALWAYS_TRUE_CONDITION, expr.location)
            else:
                self._emit(ALWAYS_FALSE_CONDITION, expr.location)

    def _check_naming_convention(
        self,
        name: str,
        kind: str,
        location: Optional[SourceLocation],
    ) -> None:
        """Check if a name follows the expected naming convention."""
        if name.startswith("_"):
            return  # Private names are allowed any convention

        # Check for snake_case (variables, functions, parameters)
        if kind in ("variable", "function", "parameter"):
            if not self._is_snake_case(name):
                self._emit(NAMING_CONVENTION, location, name, "snake_case")

        # Check for PascalCase (classes)
        elif kind == "class":
            if not self._is_pascal_case(name):
                self._emit(NAMING_CONVENTION, location, name, "PascalCase")

    def _is_snake_case(self, name: str) -> bool:
        """Check if a name is in snake_case."""
        if not name:
            return True
        # Allow single letters and all lowercase with underscores
        return bool(re.match(r"^[a-z][a-z0-9_]*$", name) or len(name) == 1)

    def _is_pascal_case(self, name: str) -> bool:
        """Check if a name is in PascalCase."""
        if not name:
            return True
        return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))


# =============================================================================
# Utility Functions
# =============================================================================


def lint_source(source: str, config: Optional[LintConfiguration] = None) -> list[LintViolation]:
    """
    Lint MathViz source code.

    This is a convenience function that parses and lints source code in one step.

    Args:
        source: MathViz source code string
        config: Optional lint configuration

    Returns:
        List of lint violations found

    Raises:
        ParseError: If the source code cannot be parsed
    """
    from mathviz.compiler.lexer import Lexer
    from mathviz.compiler.parser import Parser

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()

    linter = Linter(config)
    return linter.lint(program)


def lint_program(program: Program, config: Optional[LintConfiguration] = None) -> list[LintViolation]:
    """
    Lint an already-parsed MathViz program.

    Args:
        program: Parsed AST
        config: Optional lint configuration

    Returns:
        List of lint violations found
    """
    linter = Linter(config)
    return linter.lint(program)


def get_rule_by_name(name: str) -> Optional[LintRule]:
    """Get a lint rule by its name."""
    return RULES_BY_NAME.get(name)


def get_rule_by_code(code: str) -> Optional[LintRule]:
    """Get a lint rule by its code."""
    return ALL_RULES.get(code)


def get_rules_by_category(category: LintCategory) -> list[LintRule]:
    """Get all lint rules in a category."""
    return [rule for rule in ALL_RULES.values() if rule.category == category]


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "LintLevel",
    "LintCategory",
    # Data classes
    "LintRule",
    "LintViolation",
    "LintConfiguration",
    "SymbolInfo",
    "ScopeInfo",
    # Main linter
    "Linter",
    # Utility functions
    "lint_source",
    "lint_program",
    "get_rule_by_name",
    "get_rule_by_code",
    "get_rules_by_category",
    # Rule registry
    "ALL_RULES",
    "RULES_BY_NAME",
    # Individual rules - Unused
    "UNUSED_VARIABLE",
    "UNUSED_FUNCTION",
    "UNUSED_PARAMETER",
    "UNUSED_IMPORT",
    "UNUSED_LOOP_VARIABLE",
    # Individual rules - Unreachable
    "UNREACHABLE_CODE",
    "UNREACHABLE_PATTERN",
    "NON_EXHAUSTIVE_MATCH",
    "REDUNDANT_PATTERN",
    "REDUNDANT_ASSIGNMENT",
    "ALWAYS_TRUE_CONDITION",
    "ALWAYS_FALSE_CONDITION",
    "EMPTY_BLOCK",
    # Individual rules - Math
    "FLOAT_EQUALITY",
    "DIVISION_BY_ZERO_POSSIBLE",
    "INEFFICIENT_POWER",
    "UNNECESSARY_MATH_CALL",
    "INTEGER_DIVISION_TRUNCATION",
    # Individual rules - Style
    "NAMING_CONVENTION",
    "SHADOWING",
    "MUTABLE_DEFAULT_ARGUMENT",
    "COMPARISON_TO_NONE",
    "REDUNDANT_BOOLEAN",
    # Individual rules - Performance
    "LOOP_INVARIANT",
    "REDUNDANT_COMPUTATION",
    "INEFFICIENT_LIST_APPEND",
    "STRING_CONCAT_IN_LOOP",
    # Individual rules - Correctness
    "SELF_ASSIGNMENT",
    "COMPARISON_WITH_ITSELF",
    "UNREACHABLE_RETURN",
    "MISSING_RETURN",
]
