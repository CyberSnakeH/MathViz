"""
MathViz Interactive REPL (Read-Eval-Print Loop).

Provides an interactive shell for experimenting with MathViz code,
featuring command handling, tab completion, history, and session management.

Usage:
    mathviz repl
    mathviz -i

Example session:
    >>> let x = 10
    x = 10

    >>> x * 2
    20

    >>> fn square(n: Int) -> Int { return n ^ 2 }
    defined: square(n: Int) -> Int

    >>> :type square
    (Int) -> Int

    >>> :help
    ...
"""

from __future__ import annotations

import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import readline

    HAS_READLINE = True
except ImportError:
    # readline not available on some platforms (e.g., Windows without pyreadline)
    HAS_READLINE = False

import numpy as np

from mathviz import __version__
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.codegen import CodeGenerator
from mathviz.compiler.type_checker import TypeChecker, infer_expression_type
from mathviz.compiler.ast_nodes import (
    Program,
    FunctionDef,
    LetStatement,
    ExpressionStatement,
    Statement,
)
from mathviz.utils.errors import MathVizError


# =============================================================================
# ANSI Color Codes
# =============================================================================


class Colors:
    """ANSI escape codes for colored terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors."""
        for attr in [
            "RED",
            "GREEN",
            "YELLOW",
            "BLUE",
            "MAGENTA",
            "CYAN",
            "WHITE",
            "GRAY",
            "BOLD",
            "DIM",
            "RESET",
        ]:
            setattr(cls, attr, "")


def _init_colors() -> None:
    """Initialize colors based on terminal capabilities."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        Colors.disable()


_init_colors()


# =============================================================================
# REPL Commands
# =============================================================================


@dataclass
class REPLCommand:
    """A REPL command definition."""

    name: str
    aliases: tuple[str, ...] = ()
    help_text: str = ""
    handler: Optional[Callable[["REPLSession", str], Optional[str]]] = None


# =============================================================================
# REPL Session
# =============================================================================


@dataclass
class DefinedFunction:
    """Information about a user-defined function."""

    name: str
    params: list[str]
    param_types: list[str]
    return_type: str
    source: str


class REPLSession:
    """
    Interactive REPL session for MathViz.

    Manages the session state including variables, functions, and type environment.
    Provides command handling, evaluation, and session persistence.
    """

    def __init__(self) -> None:
        """Initialize a new REPL session."""
        # Session state
        self.variables: dict[str, Any] = {}
        self.variable_types: dict[str, str] = {}
        self.functions: dict[str, DefinedFunction] = {}
        self.history: list[str] = []

        # Execution environment
        self._globals: dict[str, Any] = {"__builtins__": __builtins__}
        self._setup_builtins()

        # Commands
        self._commands = self._setup_commands()

        # Session configuration
        self.prompt = ">>> "
        self.continuation_prompt = "... "
        self.show_types = True

    def _setup_builtins(self) -> None:
        """Setup built-in functions and constants."""
        # Mathematical constants
        self._globals.update(
            {
                "PI": math.pi,
                "E": math.e,
                "TAU": math.tau,
                "INF": float("inf"),
                "NAN": float("nan"),
            }
        )

        # Mathematical functions (numpy-backed for array support)
        self._globals.update(
            {
                "sqrt": np.sqrt,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "asin": np.arcsin,
                "acos": np.arccos,
                "atan": np.arctan,
                "atan2": np.arctan2,
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                "exp": np.exp,
                "log": np.log,
                "log10": np.log10,
                "log2": np.log2,
                "abs": np.abs,
                "floor": np.floor,
                "ceil": np.ceil,
                "round": np.round,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "print": print,
                "println": lambda *args: print(*args),
            }
        )

        # NumPy array creation
        self._globals.update(
            {
                "array": np.array,
                "zeros": np.zeros,
                "ones": np.ones,
                "linspace": np.linspace,
                "arange": np.arange,
            }
        )

        # Add variables to execution environment
        self._globals.update(self.variables)

    def _setup_commands(self) -> dict[str, REPLCommand]:
        """Setup REPL commands."""
        commands = {
            "help": REPLCommand(
                name="help",
                aliases=("h", "?"),
                help_text="Show this help message",
                handler=self._cmd_help,
            ),
            "quit": REPLCommand(
                name="quit",
                aliases=("q", "exit"),
                help_text="Exit the REPL",
                handler=self._cmd_quit,
            ),
            "clear": REPLCommand(
                name="clear",
                aliases=(),
                help_text="Clear all definitions",
                handler=self._cmd_clear,
            ),
            "type": REPLCommand(
                name="type",
                aliases=("t",),
                help_text="Show the type of an expression",
                handler=self._cmd_type,
            ),
            "ast": REPLCommand(
                name="ast",
                aliases=(),
                help_text="Show the AST of an expression",
                handler=self._cmd_ast,
            ),
            "vars": REPLCommand(
                name="vars",
                aliases=("v",),
                help_text="Show all defined variables",
                handler=self._cmd_vars,
            ),
            "funcs": REPLCommand(
                name="funcs",
                aliases=("f", "functions"),
                help_text="Show all defined functions",
                handler=self._cmd_funcs,
            ),
            "load": REPLCommand(
                name="load",
                aliases=(),
                help_text="Load and execute a .mviz file",
                handler=self._cmd_load,
            ),
            "save": REPLCommand(
                name="save",
                aliases=(),
                help_text="Save session to a file",
                handler=self._cmd_save,
            ),
            "reset": REPLCommand(
                name="reset",
                aliases=(),
                help_text="Reset the session (clear all state)",
                handler=self._cmd_reset,
            ),
        }

        # Build alias lookup
        alias_map = {}
        for cmd in commands.values():
            alias_map[cmd.name] = cmd
            for alias in cmd.aliases:
                alias_map[alias] = cmd

        return alias_map

    # -------------------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------------------

    def _cmd_help(self, session: "REPLSession", args: str) -> str:
        """Show help message."""
        lines = [
            f"{Colors.BOLD}Commands:{Colors.RESET}",
            f"  {Colors.CYAN}:help{Colors.RESET}          Show this help",
            f"  {Colors.CYAN}:quit, :q{Colors.RESET}      Exit REPL",
            f"  {Colors.CYAN}:clear{Colors.RESET}         Clear all definitions",
            f"  {Colors.CYAN}:type <expr>{Colors.RESET}   Show type of expression",
            f"  {Colors.CYAN}:ast <expr>{Colors.RESET}    Show AST of expression",
            f"  {Colors.CYAN}:vars{Colors.RESET}          Show all defined variables",
            f"  {Colors.CYAN}:funcs{Colors.RESET}         Show all defined functions",
            f"  {Colors.CYAN}:load <file>{Colors.RESET}   Load and execute a .mviz file",
            f"  {Colors.CYAN}:save <file>{Colors.RESET}   Save session to file",
            f"  {Colors.CYAN}:reset{Colors.RESET}         Reset the session",
            "",
            f"{Colors.BOLD}Syntax:{Colors.RESET}",
            f"  {Colors.GREEN}let x = 10{Colors.RESET}                  Variable declaration",
            f"  {Colors.GREEN}fn add(a: Int, b: Int) -> Int {{ return a + b }}{Colors.RESET}",
            f"  {Colors.GREEN}[x^2 for x in 0..5]{Colors.RESET}         List comprehension",
            "",
            f"{Colors.BOLD}Built-in Constants:{Colors.RESET}",
            f"  PI, E, TAU, INF, NAN",
            "",
            f"{Colors.BOLD}Built-in Functions:{Colors.RESET}",
            f"  sqrt, sin, cos, tan, exp, log, abs, floor, ceil, round",
            f"  min, max, sum, len, range, array, zeros, ones, linspace",
        ]
        return "\n".join(lines)

    def _cmd_quit(self, session: "REPLSession", args: str) -> str:
        """Exit the REPL."""
        print(f"{Colors.DIM}Goodbye!{Colors.RESET}")
        sys.exit(0)

    def _cmd_clear(self, session: "REPLSession", args: str) -> str:
        """Clear all definitions."""
        self.variables.clear()
        self.variable_types.clear()
        self.functions.clear()
        # Refresh globals
        self._globals = {"__builtins__": __builtins__}
        self._setup_builtins()
        return f"{Colors.GREEN}Cleared all definitions{Colors.RESET}"

    def _cmd_reset(self, session: "REPLSession", args: str) -> str:
        """Reset the entire session."""
        self.__init__()
        return f"{Colors.GREEN}Session reset{Colors.RESET}"

    def _cmd_type(self, session: "REPLSession", args: str) -> str:
        """Show the type of an expression."""
        if not args.strip():
            return f"{Colors.RED}Error: :type requires an expression{Colors.RESET}"

        try:
            # Parse the expression
            source = args.strip()
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()

            if not ast.statements:
                return f"{Colors.RED}Error: No expression to type-check{Colors.RESET}"

            # Get the expression from the statement
            stmt = ast.statements[0]
            if isinstance(stmt, ExpressionStatement):
                expr = stmt.expression
            else:
                return f"{Colors.YELLOW}Statement type: {type(stmt).__name__}{Colors.RESET}"

            # Infer the type
            type_result = infer_expression_type(expr, self.variable_types)
            return f"{Colors.CYAN}{type_result}{Colors.RESET}"

        except Exception as e:
            return f"{Colors.RED}Error: {e}{Colors.RESET}"

    def _cmd_ast(self, session: "REPLSession", args: str) -> str:
        """Show the AST of an expression."""
        if not args.strip():
            return f"{Colors.RED}Error: :ast requires an expression{Colors.RESET}"

        try:
            source = args.strip()
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()

            return self._format_ast(ast)

        except Exception as e:
            return f"{Colors.RED}Error: {e}{Colors.RESET}"

    def _cmd_vars(self, session: "REPLSession", args: str) -> str:
        """Show all defined variables."""
        if not self.variables:
            return f"{Colors.DIM}No variables defined{Colors.RESET}"

        lines = []
        for name, value in sorted(self.variables.items()):
            type_str = self.variable_types.get(name, "Unknown")
            lines.append(f"  {Colors.CYAN}{name}{Colors.RESET}: {type_str} = {repr(value)}")

        return "\n".join(lines)

    def _cmd_funcs(self, session: "REPLSession", args: str) -> str:
        """Show all defined functions."""
        if not self.functions:
            return f"{Colors.DIM}No functions defined{Colors.RESET}"

        lines = []
        for name, func in sorted(self.functions.items()):
            params = ", ".join(
                f"{p}: {t}" if t else p for p, t in zip(func.params, func.param_types)
            )
            ret = f" -> {func.return_type}" if func.return_type else ""
            lines.append(f"  {Colors.GREEN}{name}{Colors.RESET}({params}){ret}")

        return "\n".join(lines)

    def _cmd_load(self, session: "REPLSession", args: str) -> str:
        """Load and execute a .mviz file."""
        filepath = args.strip()
        if not filepath:
            return f"{Colors.RED}Error: :load requires a filename{Colors.RESET}"

        path = Path(filepath)
        if not path.exists():
            return f"{Colors.RED}Error: File not found: {filepath}{Colors.RESET}"

        try:
            source = path.read_text(encoding="utf-8")
            lines_executed = 0

            for line in source.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                result = self.eval_line(line)
                if result:
                    print(result)
                lines_executed += 1

            return f"{Colors.GREEN}Loaded {lines_executed} lines from {filepath}{Colors.RESET}"

        except Exception as e:
            return f"{Colors.RED}Error loading file: {e}{Colors.RESET}"

    def _cmd_save(self, session: "REPLSession", args: str) -> str:
        """Save session to a file."""
        filepath = args.strip()
        if not filepath:
            return f"{Colors.RED}Error: :save requires a filename{Colors.RESET}"

        try:
            lines = []
            lines.append(f"# MathViz REPL session")
            lines.append(f"# Saved from interactive session\n")

            # Export functions
            for name, func in self.functions.items():
                lines.append(func.source)
                lines.append("")

            # Export variables
            for name, value in self.variables.items():
                lines.append(f"let {name} = {repr(value)}")

            path = Path(filepath)
            path.write_text("\n".join(lines), encoding="utf-8")

            return f"{Colors.GREEN}Session saved to {filepath}{Colors.RESET}"

        except Exception as e:
            return f"{Colors.RED}Error saving file: {e}{Colors.RESET}"

    # -------------------------------------------------------------------------
    # AST Formatting
    # -------------------------------------------------------------------------

    def _format_ast(self, node: Any, indent: int = 0) -> str:
        """Format an AST node as a readable string."""
        prefix = "  " * indent
        node_name = type(node).__name__

        # Get relevant attributes (non-private, non-callable)
        attrs = {}
        for key in dir(node):
            if not key.startswith("_") and key not in ("accept", "location"):
                value = getattr(node, key, None)
                if not callable(value):
                    attrs[key] = value

        if not attrs:
            return f"{prefix}{Colors.YELLOW}{node_name}{Colors.RESET}"

        lines = [f"{prefix}{Colors.YELLOW}{node_name}{Colors.RESET}("]
        for key, value in attrs.items():
            if hasattr(value, "accept"):  # AST node
                lines.append(f"{prefix}  {key}=")
                lines.append(self._format_ast(value, indent + 2))
            elif isinstance(value, (list, tuple)) and value:
                if hasattr(value[0], "accept"):
                    lines.append(f"{prefix}  {key}=[")
                    for item in value:
                        lines.append(self._format_ast(item, indent + 2))
                    lines.append(f"{prefix}  ]")
                else:
                    lines.append(f"{prefix}  {key}={value!r}")
            else:
                lines.append(f"{prefix}  {key}={value!r}")

        lines.append(f"{prefix})")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def eval_line(self, line: str) -> Optional[str]:
        """
        Evaluate a single line of input.

        Returns the result string or None if no output.
        """
        line = line.strip()
        if not line:
            return None

        # Handle commands
        if line.startswith(":"):
            return self._handle_command(line)

        # Try to parse and execute
        try:
            return self._compile_and_run(line)
        except MathVizError as e:
            return f"{Colors.RED}Error: {e}{Colors.RESET}"
        except Exception as e:
            return f"{Colors.RED}Error: {e}{Colors.RESET}"

    def _handle_command(self, cmd: str) -> str:
        """Handle a REPL command."""
        parts = cmd[1:].split(maxsplit=1)
        command_name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        if command_name in self._commands:
            cmd_obj = self._commands[command_name]
            if cmd_obj.handler:
                return cmd_obj.handler(self, args) or ""
            return f"{Colors.YELLOW}Command not implemented: {command_name}{Colors.RESET}"

        return f"{Colors.RED}Unknown command: :{command_name}{Colors.RESET}\nType :help for available commands"

    def _compile_and_run(self, source: str) -> Optional[str]:
        """Compile and execute MathViz code."""
        # Parse the source
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        if not ast.statements:
            return None

        stmt = ast.statements[0]

        # Handle different statement types
        if isinstance(stmt, FunctionDef):
            return self._handle_function_def(stmt, source)
        elif isinstance(stmt, LetStatement):
            return self._handle_let_statement(stmt, source)
        elif isinstance(stmt, ExpressionStatement):
            return self._handle_expression(stmt, source)
        else:
            # General statement
            return self._execute_statement(ast, source)

    def _handle_function_def(self, func: FunctionDef, source: str) -> str:
        """Handle a function definition."""
        # Extract function info
        params = [p.name for p in func.parameters]
        param_types = [
            str(p.type_annotation.name)
            if p.type_annotation and hasattr(p.type_annotation, "name")
            else ""
            for p in func.parameters
        ]
        return_type = ""
        if func.return_type:
            if hasattr(func.return_type, "name"):
                return_type = str(func.return_type.name)

        # Store function info
        self.functions[func.name] = DefinedFunction(
            name=func.name,
            params=params,
            param_types=param_types,
            return_type=return_type,
            source=source,
        )

        # Generate and execute Python code
        codegen = CodeGenerator(optimize=False)
        python_code = codegen.generate(Program(statements=(func,)))

        # Execute the generated code in the REPL globals
        # Note: This uses Python's built-in exec() for code execution, not shell exec
        compiled = compile(python_code, "<repl>", "exec")
        eval(compiled, self._globals)  # noqa: S307 - Safe: executing compiler-generated code

        # Format output
        params_str = ", ".join(f"{p}: {t}" if t else p for p, t in zip(params, param_types))
        ret_str = f" -> {return_type}" if return_type else ""

        return f"{Colors.GREEN}defined:{Colors.RESET} {func.name}({params_str}){ret_str}"

    def _handle_let_statement(self, stmt: LetStatement, source: str) -> str:
        """Handle a let statement."""
        # Generate and execute Python code
        codegen = CodeGenerator(optimize=False)
        python_code = codegen.generate(Program(statements=(stmt,)))

        # Execute in our environment
        local_vars: dict[str, Any] = {}
        compiled = compile(python_code, "<repl>", "exec")
        eval(compiled, self._globals, local_vars)  # noqa: S307 - Safe: executing compiler-generated code

        # Extract the new variable
        if stmt.name in local_vars:
            value = local_vars[stmt.name]
            self.variables[stmt.name] = value
            self._globals[stmt.name] = value

            # Determine type
            type_str = self._infer_value_type(value)
            if stmt.type_annotation and hasattr(stmt.type_annotation, "name"):
                type_str = str(stmt.type_annotation.name)
            self.variable_types[stmt.name] = type_str

            return f"{Colors.CYAN}{stmt.name}{Colors.RESET} = {repr(value)}"

        return ""

    def _handle_expression(self, stmt: ExpressionStatement, source: str) -> Optional[str]:
        """Handle an expression statement."""
        # Wrap in a result assignment to capture the value
        wrapped_source = f"__repl_result__ = ({source})"

        try:
            lexer = Lexer(wrapped_source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()

            codegen = CodeGenerator(optimize=False)
            python_code = codegen.generate(ast)

            local_vars: dict[str, Any] = {}
            compiled = compile(python_code, "<repl>", "exec")
            eval(compiled, self._globals, local_vars)  # noqa: S307 - Safe: executing compiler-generated code

            if "__repl_result__" in local_vars:
                result = local_vars["__repl_result__"]
                if result is not None:
                    return repr(result)

        except Exception:
            # Fall back to direct execution (might be a statement with side effects)
            codegen = CodeGenerator(optimize=False)
            ast = Program(statements=(stmt,))
            python_code = codegen.generate(ast)
            compiled = compile(python_code, "<repl>", "exec")
            eval(compiled, self._globals)  # noqa: S307 - Safe: executing compiler-generated code

        return None

    def _execute_statement(self, ast: Program, source: str) -> Optional[str]:
        """Execute a general statement."""
        codegen = CodeGenerator(optimize=False)
        python_code = codegen.generate(ast)
        compiled = compile(python_code, "<repl>", "exec")
        eval(compiled, self._globals)  # noqa: S307 - Safe: executing compiler-generated code
        return None

    def _infer_value_type(self, value: Any) -> str:
        """Infer the type string for a runtime value."""
        if isinstance(value, bool):
            return "Bool"
        elif isinstance(value, int):
            return "Int"
        elif isinstance(value, float):
            return "Float"
        elif isinstance(value, str):
            return "String"
        elif isinstance(value, list):
            if value:
                elem_type = self._infer_value_type(value[0])
                return f"List[{elem_type}]"
            return "List[Unknown]"
        elif isinstance(value, tuple):
            if value:
                types = [self._infer_value_type(v) for v in value]
                return f"({', '.join(types)})"
            return "()"
        elif isinstance(value, set):
            if value:
                elem_type = self._infer_value_type(next(iter(value)))
                return f"Set[{elem_type}]"
            return "Set[Unknown]"
        elif isinstance(value, dict):
            if value:
                key = next(iter(value))
                key_type = self._infer_value_type(key)
                val_type = self._infer_value_type(value[key])
                return f"Dict[{key_type}, {val_type}]"
            return "Dict[Unknown, Unknown]"
        elif isinstance(value, np.ndarray):
            return f"Array[{value.dtype}]"
        elif value is None:
            return "None"
        else:
            return type(value).__name__

    def _is_incomplete(self, line: str) -> bool:
        """Check if a line is incomplete (needs continuation)."""
        # Count open brackets/braces/parens
        open_count = line.count("{") - line.count("}")
        open_count += line.count("[") - line.count("]")
        open_count += line.count("(") - line.count(")")
        return open_count > 0

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Main REPL loop."""
        print(f"{Colors.BOLD}MathViz {__version__}{Colors.RESET} - Interactive Mode")
        print(
            f"Type {Colors.CYAN}:help{Colors.RESET} for help, {Colors.CYAN}:quit{Colors.RESET} to exit"
        )
        print()

        # Setup readline if available
        if HAS_READLINE:
            completer = REPLCompleter(self)
            readline.set_completer(completer.complete)
            readline.parse_and_bind("tab: complete")

            # Load history
            history_file = Path.home() / ".mathviz_history"
            try:
                if history_file.exists():
                    readline.read_history_file(str(history_file))
            except Exception:
                pass

        try:
            while True:
                try:
                    line = input(self.prompt)

                    # Handle multi-line input
                    while line.rstrip().endswith("{") or self._is_incomplete(line):
                        continuation = input(self.continuation_prompt)
                        line += "\n" + continuation

                    # Store in history
                    self.history.append(line)

                    # Evaluate and print result
                    result = self.eval_line(line)
                    if result:
                        print(result)

                except KeyboardInterrupt:
                    print(f"\n{Colors.DIM}Use :quit to exit{Colors.RESET}")
                except EOFError:
                    print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}")
                    break

        finally:
            # Save history
            if HAS_READLINE:
                history_file = Path.home() / ".mathviz_history"
                try:
                    readline.set_history_length(1000)
                    readline.write_history_file(str(history_file))
                except Exception:
                    pass


# =============================================================================
# Tab Completion
# =============================================================================


class REPLCompleter:
    """Tab completion for the REPL."""

    def __init__(self, session: REPLSession) -> None:
        self.session = session

        # Keywords and built-in constructs
        self.keywords = [
            "let",
            "fn",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "return",
            "break",
            "continue",
            "pass",
            "match",
            "struct",
            "enum",
            "trait",
            "impl",
            "pub",
            "true",
            "false",
            "None",
            "in",
            "and",
            "or",
            "not",
            "Some",
            "Ok",
            "Err",
        ]

        # Built-in constants
        self.constants = ["PI", "E", "TAU", "INF", "NAN"]

        # Built-in functions
        self.builtin_functions = [
            "sqrt",
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
            "exp",
            "log",
            "log10",
            "log2",
            "abs",
            "floor",
            "ceil",
            "round",
            "min",
            "max",
            "sum",
            "len",
            "range",
            "print",
            "println",
            "array",
            "zeros",
            "ones",
            "linspace",
            "arange",
        ]

        # REPL commands
        self.commands = [
            ":help",
            ":quit",
            ":q",
            ":clear",
            ":type",
            ":ast",
            ":vars",
            ":funcs",
            ":load",
            ":save",
            ":reset",
        ]

    def complete(self, text: str, state: int) -> Optional[str]:
        """Get completions for the given text."""
        if state == 0:
            # Build completions on first call
            self._completions = self._get_completions(text)
        try:
            return self._completions[state]
        except IndexError:
            return None

    def _get_completions(self, text: str) -> list[str]:
        """Get all completions for the given text prefix."""
        completions = []

        # Command completions
        if text.startswith(":"):
            completions.extend(c for c in self.commands if c.startswith(text))
            return sorted(set(completions))

        # Variable completions
        completions.extend(name for name in self.session.variables if name.startswith(text))

        # Function completions
        completions.extend(name for name in self.session.functions if name.startswith(text))

        # Keyword completions
        completions.extend(kw for kw in self.keywords if kw.startswith(text))

        # Constant completions
        completions.extend(c for c in self.constants if c.startswith(text))

        # Built-in function completions
        completions.extend(f for f in self.builtin_functions if f.startswith(text))

        return sorted(set(completions))


# =============================================================================
# Entry Point
# =============================================================================


def main() -> int:
    """Entry point for the REPL."""
    session = REPLSession()
    session.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
