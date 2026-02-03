"""
Purity Analysis Module for MathViz Compiler.

This module analyzes functions to determine their purity characteristics,
identifying side effects, I/O operations, and Manim-specific calls. This
information is used by the JIT optimizer to determine which functions
can be safely optimized without changing program semantics.

A pure function has no side effects and always produces the same output
for the same inputs. Impure functions may:
- Read or write global variables
- Mutate parameters or external state
- Perform I/O operations (print, file operations)
- Execute Manim animation calls (play, wait)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from mathviz.compiler.ast_nodes import (
    ASTNode,
    AssignmentStatement,
    BaseASTVisitor,
    Block,
    CallExpression,
    CompoundAssignment,
    ConditionalExpression,
    Expression,
    ExpressionStatement,
    ForStatement,
    FunctionDef,
    Identifier,
    IfStatement,
    IndexExpression,
    LambdaExpression,
    LetStatement,
    MemberAccess,
    Parameter,
    PlayStatement,
    PrintStatement,
    Program,
    ReturnStatement,
    SceneDef,
    WaitStatement,
    WhileStatement,
)
from mathviz.utils.errors import SourceLocation


class Purity(Enum):
    """
    Classification of function purity levels.

    Functions are classified from most restrictive (PURE) to least restrictive.
    A function's purity is determined by the most impure operation it performs.
    """

    PURE = auto()
    """
    Completely pure function - no side effects, deterministic output.
    These functions are safe for memoization, parallelization, and aggressive
    inlining optimizations.
    """

    IMPURE_IO = auto()
    """
    Function performs I/O operations (print, file read/write).
    These cannot be memoized or reordered but may still be parallelized
    in some contexts.
    """

    IMPURE_MUTATION = auto()
    """
    Function mutates state (global variables, parameters).
    These functions require careful ordering and cannot be safely
    parallelized without synchronization.
    """

    IMPURE_MANIM = auto()
    """
    Function performs Manim animation operations (play, wait).
    These have strict sequential ordering requirements and interact
    with the rendering system.
    """


class SideEffectKind(Enum):
    """Types of side effects that can be detected."""

    GLOBAL_READ = auto()
    """Reading from a global variable."""

    GLOBAL_WRITE = auto()
    """Writing to a global variable."""

    PARAMETER_MUTATION = auto()
    """Mutating a function parameter (e.g., array modification)."""

    IO_PRINT = auto()
    """Print or println operation."""

    IO_FILE_READ = auto()
    """File read operation."""

    IO_FILE_WRITE = auto()
    """File write operation."""

    MANIM_PLAY = auto()
    """Manim play animation call."""

    MANIM_WAIT = auto()
    """Manim wait call."""

    MANIM_SCENE_METHOD = auto()
    """Generic Manim Scene method call."""

    EXTERNAL_CALL = auto()
    """Call to an external function with unknown purity."""


@dataclass(frozen=True, slots=True)
class SideEffect:
    """
    Represents a detected side effect in a function.

    Attributes:
        kind: The type of side effect.
        name: The identifier associated with the side effect (variable name,
              function name, etc.).
        location: Source location where the side effect occurs.
        description: Human-readable description of the side effect.
    """

    kind: SideEffectKind
    name: str
    location: Optional[SourceLocation] = None
    description: str = ""

    def __str__(self) -> str:
        loc_str = f" at {self.location}" if self.location else ""
        desc_str = f": {self.description}" if self.description else ""
        return f"{self.kind.name}({self.name}){loc_str}{desc_str}"


@dataclass
class PurityInfo:
    """
    Complete purity analysis result for a function.

    This dataclass contains all information gathered during purity analysis,
    including the overall purity classification and detailed lists of
    detected side effects.
    """

    purity: Purity = Purity.PURE
    """Overall purity classification."""

    side_effects: list[SideEffect] = field(default_factory=list)
    """All detected side effects."""

    io_calls: list[str] = field(default_factory=list)
    """Names of I/O functions called (print, read_file, write_file, etc.)."""

    manim_calls: list[str] = field(default_factory=list)
    """Names of Manim functions called (play, wait, Scene methods)."""

    reads_globals: set[str] = field(default_factory=set)
    """Names of global variables read by the function."""

    writes_globals: set[str] = field(default_factory=set)
    """Names of global variables written by the function."""

    def is_pure(self) -> bool:
        """Check if the function is completely pure."""
        return self.purity == Purity.PURE

    def has_io(self) -> bool:
        """Check if the function performs any I/O operations."""
        return bool(self.io_calls)

    def has_manim_calls(self) -> bool:
        """Check if the function performs any Manim operations."""
        return bool(self.manim_calls)

    def has_mutations(self) -> bool:
        """Check if the function mutates any state."""
        return bool(self.writes_globals) or any(
            se.kind == SideEffectKind.PARAMETER_MUTATION for se in self.side_effects
        )

    def merge(self, other: "PurityInfo") -> "PurityInfo":
        """
        Merge two PurityInfo instances, taking the worst purity level.

        This is useful for combining analysis results from nested functions
        or conditional branches.
        """
        # Determine worst purity (higher enum value = less pure)
        worst_purity = max(self.purity, other.purity, key=lambda p: p.value)

        return PurityInfo(
            purity=worst_purity,
            side_effects=self.side_effects + other.side_effects,
            io_calls=self.io_calls + other.io_calls,
            manim_calls=self.manim_calls + other.manim_calls,
            reads_globals=self.reads_globals | other.reads_globals,
            writes_globals=self.writes_globals | other.writes_globals,
        )


# Known pure functions that can be safely optimized
KNOWN_PURE_FUNCTIONS: frozenset[str] = frozenset(
    {
        # Mathematical functions
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "sqrt",
        "exp",
        "log",
        "log2",
        "log10",
        "pow",
        "abs",
        "floor",
        "ceil",
        "round",
        "min",
        "max",
        "sum",
        "prod",
        "factorial",
        "gcd",
        "lcm",
        # Type conversions
        "int",
        "float",
        "str",
        "bool",
        "len",
        "range",
        # Array/collection operations that return new values
        "map",
        "filter",
        "reduce",
        "zip",
        "enumerate",
        "sorted",
        "reversed",
        "list",
        "tuple",
        "set",
        "dict",
        "frozenset",
        # String operations
        "format",
        "repr",
        "chr",
        "ord",
        # Mathematical constants
        "pi",
        "e",
        "tau",
        "inf",
        # NumPy pure operations (return new arrays)
        "np.sin",
        "np.cos",
        "np.tan",
        "np.sqrt",
        "np.exp",
        "np.log",
        "np.abs",
        "np.sum",
        "np.mean",
        "np.std",
        "np.var",
        "np.min",
        "np.max",
        "np.dot",
        "np.cross",
        "np.zeros",
        "np.ones",
        "np.array",
        "np.linspace",
        "np.arange",
        "np.reshape",
        "np.transpose",
        "np.concatenate",
        "np.stack",
        "np.vstack",
        "np.hstack",
    }
)

# I/O functions that have side effects
IO_FUNCTIONS: frozenset[str] = frozenset(
    {
        "print",
        "println",
        "read_file",
        "write_file",
        "open",
        "input",
        "read",
        "write",
        "readlines",
        "writelines",
    }
)

# Manim-specific functions and methods
MANIM_FUNCTIONS: frozenset[str] = frozenset(
    {
        "play",
        "wait",
        "add",
        "remove",
        "bring_to_front",
        "bring_to_back",
        "clear",
        "add_foreground_mobject",
        "add_foreground_mobjects",
        "remove_foreground_mobject",
        "remove_foreground_mobjects",
        "add_updater",
        "remove_updater",
        "set_camera_orientation",
        "begin_ambient_camera_rotation",
        "stop_ambient_camera_rotation",
        "move_camera",
    }
)

# Methods that indicate self.* Manim Scene calls
MANIM_SELF_METHODS: frozenset[str] = frozenset(
    {
        "play",
        "wait",
        "add",
        "remove",
        "bring_to_front",
        "bring_to_back",
        "clear",
        "next_section",
        "start_loop",
        "end_loop",
    }
)


class PurityAnalyzer(BaseASTVisitor):
    """
    Analyzes MathViz functions for purity characteristics.

    This visitor traverses function bodies to identify side effects,
    I/O operations, and Manim calls. The analysis is conservative:
    any operation that might have side effects is flagged.

    Usage:
        analyzer = PurityAnalyzer()
        purity_info = analyzer.analyze_function(func_def)

        # Or analyze an entire program
        all_purity = analyzer.analyze_program(program)
    """

    def __init__(self) -> None:
        """Initialize the purity analyzer."""
        self._reset_state()
        # Track function definitions for inter-procedural analysis
        self._function_purity: dict[str, PurityInfo] = {}
        # Track which functions are currently being analyzed (for recursion)
        self._analyzing: set[str] = set()

    def _reset_state(self) -> None:
        """Reset analysis state for a new function."""
        self._current_info = PurityInfo()
        self._local_variables: set[str] = set()
        self._parameters: set[str] = set()
        self._in_scene_context: bool = False

    def analyze_function(self, func: FunctionDef) -> PurityInfo:
        """
        Analyze a function for purity.

        Args:
            func: The function definition AST node to analyze.

        Returns:
            PurityInfo containing the complete purity analysis.
        """
        # Check for recursive analysis
        if func.name in self._analyzing:
            # Conservative assumption for recursive functions
            return PurityInfo(purity=Purity.IMPURE_MUTATION)

        self._analyzing.add(func.name)
        self._reset_state()

        # Register parameters as known local names
        for param in func.parameters:
            self._parameters.add(param.name)

        # Analyze the function body
        if func.body:
            self.visit(func.body)

        # Cache result for inter-procedural analysis
        result = self._current_info
        self._function_purity[func.name] = result
        self._analyzing.discard(func.name)

        return result

    def analyze_program(self, program: Program) -> dict[str, PurityInfo]:
        """
        Analyze all functions in a program.

        Args:
            program: The program AST root node.

        Returns:
            Dictionary mapping function names to their PurityInfo.
        """
        result: dict[str, PurityInfo] = {}

        # First pass: collect all function definitions
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                self._function_purity[stmt.name] = PurityInfo()
            elif isinstance(stmt, SceneDef):
                # Scene definitions are inherently impure (Manim)
                result[stmt.name] = PurityInfo(
                    purity=Purity.IMPURE_MANIM,
                    side_effects=[
                        SideEffect(
                            kind=SideEffectKind.MANIM_SCENE_METHOD,
                            name=stmt.name,
                            location=stmt.location,
                            description="Scene definition",
                        )
                    ],
                    manim_calls=["Scene"],
                )

        # Second pass: analyze each function
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                result[stmt.name] = self.analyze_function(stmt)
            elif isinstance(stmt, SceneDef) and stmt.body:
                # Analyze scene body with scene context
                self._reset_state()
                self._in_scene_context = True
                self.visit(stmt.body)
                scene_info = self._current_info
                # Merge with base scene impurity
                result[stmt.name] = result[stmt.name].merge(scene_info)

        return result

    # -------------------------------------------------------------------------
    # Statement Visitors
    # -------------------------------------------------------------------------

    def visit_let_statement(self, node: LetStatement) -> None:
        """Track local variable declarations."""
        self._local_variables.add(node.name)
        if node.value:
            self.visit(node.value)

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        """Detect global variable writes and parameter mutations."""
        self.visit(node.value)
        self._analyze_assignment_target(node.target, node.location)

    def visit_compound_assignment(self, node: CompoundAssignment) -> None:
        """Detect compound assignment side effects."""
        self.visit(node.value)
        self._analyze_assignment_target(node.target, node.location)

    def _analyze_assignment_target(
        self, target: Expression, location: Optional[SourceLocation]
    ) -> None:
        """
        Analyze an assignment target for side effects.

        Args:
            target: The expression being assigned to.
            location: Source location for error reporting.
        """
        if isinstance(target, Identifier):
            name = target.name
            if name not in self._local_variables and name not in self._parameters:
                # Writing to a global variable
                self._add_side_effect(
                    SideEffectKind.GLOBAL_WRITE,
                    name,
                    location,
                    f"Assignment to global variable '{name}'",
                )
                self._current_info.writes_globals.add(name)
                self._upgrade_purity(Purity.IMPURE_MUTATION)
        elif isinstance(target, IndexExpression):
            # Array/dict mutation
            base = self._get_base_identifier(target.object)
            if base:
                if base in self._parameters:
                    self._add_side_effect(
                        SideEffectKind.PARAMETER_MUTATION,
                        base,
                        location,
                        f"Mutation of parameter '{base}'",
                    )
                    self._upgrade_purity(Purity.IMPURE_MUTATION)
                elif base not in self._local_variables:
                    self._add_side_effect(
                        SideEffectKind.GLOBAL_WRITE,
                        base,
                        location,
                        f"Mutation of global '{base}'",
                    )
                    self._current_info.writes_globals.add(base)
                    self._upgrade_purity(Purity.IMPURE_MUTATION)
            self.visit(target.index)
        elif isinstance(target, MemberAccess):
            # Member assignment could be mutation
            base = self._get_base_identifier(target.object)
            if base:
                if base in self._parameters:
                    self._add_side_effect(
                        SideEffectKind.PARAMETER_MUTATION,
                        base,
                        location,
                        f"Mutation of parameter '{base}' member '{target.member}'",
                    )
                    self._upgrade_purity(Purity.IMPURE_MUTATION)
                elif base not in self._local_variables and base != "self":
                    self._add_side_effect(
                        SideEffectKind.GLOBAL_WRITE,
                        base,
                        location,
                        f"Mutation of global '{base}' member '{target.member}'",
                    )
                    self._current_info.writes_globals.add(base)
                    self._upgrade_purity(Purity.IMPURE_MUTATION)

    def visit_print_statement(self, node: PrintStatement) -> None:
        """Detect print statements as I/O side effects."""
        call_name = "println" if node.newline else "print"
        self._add_side_effect(
            SideEffectKind.IO_PRINT,
            call_name,
            node.location,
            f"Output operation: {call_name}",
        )
        self._current_info.io_calls.append(call_name)
        self._upgrade_purity(Purity.IMPURE_IO)

        # Still analyze the arguments
        self.visit(node.format_string)
        for arg in node.arguments:
            self.visit(arg)

    def visit_play_statement(self, node: PlayStatement) -> None:
        """Detect Manim play statements."""
        self._add_side_effect(
            SideEffectKind.MANIM_PLAY,
            "play",
            node.location,
            "Manim animation play",
        )
        self._current_info.manim_calls.append("play")
        self._upgrade_purity(Purity.IMPURE_MANIM)

        self.visit(node.animation)
        if node.run_time:
            self.visit(node.run_time)

    def visit_wait_statement(self, node: WaitStatement) -> None:
        """Detect Manim wait statements."""
        self._add_side_effect(
            SideEffectKind.MANIM_WAIT,
            "wait",
            node.location,
            "Manim wait operation",
        )
        self._current_info.manim_calls.append("wait")
        self._upgrade_purity(Purity.IMPURE_MANIM)

        if node.duration:
            self.visit(node.duration)

    def visit_for_statement(self, node: ForStatement) -> None:
        """Track loop variable as local."""
        self._local_variables.add(node.variable)
        self.visit(node.iterable)
        self.visit(node.body)

    # -------------------------------------------------------------------------
    # Expression Visitors
    # -------------------------------------------------------------------------

    def visit_identifier(self, node: Identifier) -> None:
        """Detect global variable reads."""
        name = node.name
        if (
            name not in self._local_variables
            and name not in self._parameters
            and name not in KNOWN_PURE_FUNCTIONS
            and name != "self"
        ):
            # Reading from a potentially global variable
            self._current_info.reads_globals.add(name)
            self._add_side_effect(
                SideEffectKind.GLOBAL_READ,
                name,
                node.location,
                f"Read from global variable '{name}'",
            )
            # Note: reading globals doesn't upgrade purity by itself,
            # but it does affect determinism

    def visit_call_expression(self, node: CallExpression) -> None:
        """Detect function calls and classify their purity."""
        callee = node.callee
        call_name = self._get_call_name(callee)

        if call_name:
            self._analyze_call(call_name, node.location, callee)

        # Visit callee and arguments
        self.visit(callee)
        for arg in node.arguments:
            self.visit(arg)

    def _get_call_name(self, callee: Expression) -> Optional[str]:
        """Extract the function name from a callee expression."""
        if isinstance(callee, Identifier):
            return callee.name
        elif isinstance(callee, MemberAccess):
            base = self._get_base_identifier(callee.object)
            if base:
                return f"{base}.{callee.member}"
            return callee.member
        return None

    def _get_base_identifier(self, expr: Expression) -> Optional[str]:
        """Get the base identifier from a potentially nested expression."""
        if isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, MemberAccess):
            return self._get_base_identifier(expr.object)
        elif isinstance(expr, IndexExpression):
            return self._get_base_identifier(expr.object)
        return None

    def _analyze_call(
        self,
        call_name: str,
        location: Optional[SourceLocation],
        callee: Expression,
    ) -> None:
        """
        Analyze a function call for side effects.

        Args:
            call_name: The name of the function being called.
            location: Source location of the call.
            callee: The callee expression for member access detection.
        """
        # Check for I/O functions
        base_name = call_name.split(".")[-1] if "." in call_name else call_name

        if base_name in IO_FUNCTIONS:
            kind = self._get_io_kind(base_name)
            self._add_side_effect(kind, call_name, location, f"I/O operation: {call_name}")
            self._current_info.io_calls.append(call_name)
            self._upgrade_purity(Purity.IMPURE_IO)
            return

        # Check for Manim functions
        if base_name in MANIM_FUNCTIONS:
            kind = self._get_manim_kind(base_name)
            self._add_side_effect(kind, call_name, location, f"Manim operation: {call_name}")
            self._current_info.manim_calls.append(call_name)
            self._upgrade_purity(Purity.IMPURE_MANIM)
            return

        # Check for self.* Manim methods in scene context
        if isinstance(callee, MemberAccess):
            base = self._get_base_identifier(callee.object)
            if base == "self" and callee.member in MANIM_SELF_METHODS:
                kind = self._get_manim_kind(callee.member)
                self._add_side_effect(
                    kind,
                    f"self.{callee.member}",
                    location,
                    f"Manim Scene method: self.{callee.member}",
                )
                self._current_info.manim_calls.append(f"self.{callee.member}")
                self._upgrade_purity(Purity.IMPURE_MANIM)
                return

        # Check if this is a known pure function
        if call_name in KNOWN_PURE_FUNCTIONS:
            return  # No side effect

        # Check if we've analyzed this function before
        if call_name in self._function_purity:
            called_purity = self._function_purity[call_name]
            if not called_purity.is_pure():
                # Propagate impurity
                self._current_info = self._current_info.merge(called_purity)
            return

        # Unknown function - conservative approach
        self._add_side_effect(
            SideEffectKind.EXTERNAL_CALL,
            call_name,
            location,
            f"Call to external function '{call_name}' with unknown purity",
        )
        # Don't upgrade purity for unknown calls - be optimistic
        # The JIT optimizer can handle this conservatively

    def _get_io_kind(self, name: str) -> SideEffectKind:
        """Map I/O function name to side effect kind."""
        if name in ("print", "println"):
            return SideEffectKind.IO_PRINT
        elif name in ("read_file", "read", "readlines", "open", "input"):
            return SideEffectKind.IO_FILE_READ
        else:
            return SideEffectKind.IO_FILE_WRITE

    def _get_manim_kind(self, name: str) -> SideEffectKind:
        """Map Manim function name to side effect kind."""
        if name == "play":
            return SideEffectKind.MANIM_PLAY
        elif name == "wait":
            return SideEffectKind.MANIM_WAIT
        else:
            return SideEffectKind.MANIM_SCENE_METHOD

    def visit_lambda_expression(self, node: LambdaExpression) -> None:
        """
        Analyze lambda expressions.

        Lambdas create a new scope, but can still capture and mutate
        outer variables (closures).
        """
        # Save current local variables
        outer_locals = self._local_variables.copy()
        outer_params = self._parameters.copy()

        # Add lambda parameters
        for param in node.parameters:
            self._parameters.add(param.name)

        # Analyze lambda body
        self.visit(node.body)

        # Restore scope
        self._local_variables = outer_locals
        self._parameters = outer_params

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _add_side_effect(
        self,
        kind: SideEffectKind,
        name: str,
        location: Optional[SourceLocation],
        description: str = "",
    ) -> None:
        """Add a side effect to the current analysis."""
        self._current_info.side_effects.append(
            SideEffect(kind=kind, name=name, location=location, description=description)
        )

    def _upgrade_purity(self, new_purity: Purity) -> None:
        """
        Upgrade (downgrade) purity to a less pure level if necessary.

        Purity can only get worse, never better.
        """
        if new_purity.value > self._current_info.purity.value:
            self._current_info.purity = new_purity

    def get_known_pure_functions(self) -> frozenset[str]:
        """Return the set of known pure functions."""
        return KNOWN_PURE_FUNCTIONS

    def register_pure_function(self, name: str) -> None:
        """
        Register a user-defined function as pure.

        This allows users to mark their functions as pure for optimization.
        Note: This modifies module-level state.
        """
        global KNOWN_PURE_FUNCTIONS
        KNOWN_PURE_FUNCTIONS = KNOWN_PURE_FUNCTIONS | frozenset({name})

    def is_function_pure(self, func_name: str) -> bool:
        """
        Check if a function is known to be pure.

        Args:
            func_name: Name of the function to check.

        Returns:
            True if the function is known pure, False otherwise.
        """
        if func_name in KNOWN_PURE_FUNCTIONS:
            return True
        if func_name in self._function_purity:
            return self._function_purity[func_name].is_pure()
        return False


def analyze_purity(program: Program) -> dict[str, PurityInfo]:
    """
    Convenience function to analyze all functions in a program.

    Args:
        program: The program AST to analyze.

    Returns:
        Dictionary mapping function names to their purity information.
    """
    analyzer = PurityAnalyzer()
    return analyzer.analyze_program(program)


def is_jit_safe(purity_info: PurityInfo) -> bool:
    """
    Determine if a function is safe for JIT optimization.

    A function is JIT-safe if it:
    - Is pure, OR
    - Only has I/O side effects (can still be JIT compiled, just not memoized)

    Manim operations and mutations make JIT optimization unsafe.

    Args:
        purity_info: The purity analysis result for the function.

    Returns:
        True if the function can be safely JIT compiled.
    """
    return purity_info.purity in (Purity.PURE, Purity.IMPURE_IO)


def can_memoize(purity_info: PurityInfo) -> bool:
    """
    Determine if a function's results can be safely memoized.

    Only pure functions can be memoized, as their output depends
    solely on their inputs.

    Args:
        purity_info: The purity analysis result for the function.

    Returns:
        True if the function can be memoized.
    """
    return purity_info.is_pure() and not purity_info.reads_globals


def can_parallelize(purity_info: PurityInfo) -> bool:
    """
    Determine if a function can be safely parallelized.

    Functions with mutations or Manim calls cannot be parallelized
    as they have ordering dependencies.

    Args:
        purity_info: The purity analysis result for the function.

    Returns:
        True if the function can be parallelized.
    """
    return purity_info.purity in (Purity.PURE, Purity.IMPURE_IO)
