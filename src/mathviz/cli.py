"""
MathViz Command-Line Interface.

Provides commands to compile and run MathViz programs.

Usage:
    mathviz compile input.mviz -o output.py
    mathviz run input.mviz
    mathviz check input.mviz
    mathviz analyze input.mviz
    mathviz typecheck input.mviz
    mathviz repl                    # Interactive mode
    mathviz fmt input.mviz          # Format code
    mathviz watch input.mviz        # Watch and recompile
    mathviz doc src/                # Generate docs
    mathviz init myproject          # Create new project
    mathviz info                    # Show language info
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, Optional

from mathviz import __version__
from mathviz.compiler import compile_source, compile_file
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.type_checker import TypeChecker, type_check
from mathviz.compiler.purity_analyzer import PurityAnalyzer, Purity, analyze_purity, is_jit_safe
from mathviz.compiler.complexity_analyzer import ComplexityAnalyzer, Complexity, analyze_complexity
from mathviz.compiler.call_graph import CallGraphBuilder, CallGraph
from mathviz.compiler.parallel_analyzer import ParallelAnalyzer, analyze_parallelization
from mathviz.compiler.codegen import CodeGenerator
from mathviz.compiler.linter import (
    Linter,
    LintConfiguration,
    LintLevel,
    LintCategory,
    LintViolation,
    ALL_RULES,
    RULES_BY_NAME,
)
from mathviz.utils.errors import MathVizError, TypeError as MathVizTypeError


# =============================================================================
# ANSI Color Codes for Terminal Output
# =============================================================================


class Colors:
    """ANSI escape codes for colored terminal output."""

    # Text colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Reset
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors (for non-TTY output)."""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.GRAY = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.UNDERLINE = ""
        cls.RESET = ""


def _init_colors() -> None:
    """Initialize colors based on terminal capabilities."""
    # Disable colors if not a TTY or if NO_COLOR is set
    import os

    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        Colors.disable()


# Initialize on module load
_init_colors()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mathviz",
        description="MathViz - A domain-specific language for mathematical animations",
        epilog="For more information, visit https://github.com/CyberSnakeH/MathViz",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compile command
    compile_parser = subparsers.add_parser(
        "compile",
        aliases=["c"],
        help="Compile a MathViz file to Python",
    )
    compile_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )
    compile_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Python file (default: input file with .py extension)",
    )
    compile_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable Numba JIT optimization",
    )
    compile_parser.add_argument(
        "--no-typecheck",
        action="store_true",
        help="Skip type checking during compilation",
    )
    compile_parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable auto-parallelization of loops",
    )
    compile_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat type warnings as errors (fail on any type issue)",
    )
    compile_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis information during compilation",
    )
    compile_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated code to stdout instead of file",
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        aliases=["r"],
        help="Compile and run a MathViz file",
    )
    run_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )
    run_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable Numba JIT optimization",
    )
    run_parser.add_argument(
        "-q",
        "--quality",
        choices=["l", "m", "h", "p", "k"],
        default="m",
        help="Manim render quality (l=low, m=medium, h=high, p=production, k=4k)",
    )
    run_parser.add_argument(
        "--preview",
        action="store_true",
        help="Open preview after rendering",
    )

    # Check command (syntax validation)
    check_parser = subparsers.add_parser(
        "check",
        help="Check a MathViz file for syntax errors",
    )
    check_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )

    # Analyze command (comprehensive analysis)
    analyze_parser = subparsers.add_parser(
        "analyze",
        aliases=["a"],
        help="Run comprehensive analysis on a MathViz file",
    )
    analyze_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis results as JSON",
    )

    # Typecheck command
    typecheck_parser = subparsers.add_parser(
        "typecheck",
        aliases=["tc"],
        help="Type check a MathViz file",
    )
    typecheck_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )
    typecheck_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    # Tokens command (debug)
    tokens_parser = subparsers.add_parser(
        "tokens",
        help="Show tokens for a MathViz file (debug)",
    )
    tokens_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )

    # AST command (debug)
    ast_parser = subparsers.add_parser(
        "ast",
        help="Show AST for a MathViz file (debug)",
    )
    ast_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )

    # New command (create project)
    new_parser = subparsers.add_parser(
        "new",
        help="Create a new MathViz project",
    )
    new_parser.add_argument(
        "name",
        type=str,
        help="Project name",
    )
    new_parser.add_argument(
        "--template",
        choices=["basic", "manim", "math"],
        default="basic",
        help="Project template (default: basic)",
    )

    # Exec command (run without manim)
    exec_parser = subparsers.add_parser(
        "exec",
        aliases=["e"],
        help="Compile and execute a MathViz file (without Manim)",
    )
    exec_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file (.mviz)",
    )
    exec_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable Numba JIT optimization",
    )

    # Lint command
    lint_parser = subparsers.add_parser(
        "lint",
        aliases=["l"],
        help="Run linter on a MathViz file",
    )
    lint_parser.add_argument(
        "input",
        type=Path,
        nargs="?",  # Make optional to allow --list-rules without input
        default=None,
        help="Input MathViz file (.mviz)",
    )
    lint_parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix what can be fixed (not yet implemented)",
    )
    lint_parser.add_argument(
        "--warn-all",
        action="store_true",
        help="Enable all warnings",
    )
    lint_parser.add_argument(
        "--deny",
        type=str,
        metavar="CATEGORY",
        help="Treat warnings in CATEGORY as errors (unused, unreachable, style, math, performance, correctness)",
    )
    lint_parser.add_argument(
        "--allow",
        type=str,
        metavar="RULE",
        help="Disable a specific rule (e.g., 'unused-variable' or 'W0001')",
    )
    lint_parser.add_argument(
        "--json",
        action="store_true",
        help="Output lint results as JSON",
    )
    lint_parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List all available lint rules",
    )

    # REPL command
    repl_parser = subparsers.add_parser(
        "repl",
        aliases=["i"],
        help="Start interactive REPL mode",
    )

    # Format command
    fmt_parser = subparsers.add_parser(
        "fmt",
        aliases=["format"],
        help="Format MathViz source files",
    )
    fmt_parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input file or directory (default: current directory)",
    )
    fmt_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are formatted without modifying them",
    )
    fmt_parser.add_argument(
        "--diff",
        action="store_true",
        help="Show diff of formatting changes",
    )
    fmt_parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Write changes to files (default: print to stdout)",
    )

    # Watch command
    watch_parser = subparsers.add_parser(
        "watch",
        aliases=["w"],
        help="Watch files and recompile on changes",
    )
    watch_parser.add_argument(
        "input",
        type=Path,
        help="Input MathViz file or directory to watch",
    )
    watch_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Python file (default: input file with .py extension)",
    )
    watch_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable Numba JIT optimization",
    )

    # Doc command
    doc_parser = subparsers.add_parser(
        "doc",
        help="Generate documentation from source files",
    )
    doc_parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input file or directory (default: current directory)",
    )
    doc_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for generated docs",
    )
    doc_parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML documentation",
    )
    doc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output documentation as JSON",
    )

    # Init command (enhanced version of 'new')
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new MathViz project",
    )
    init_parser.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        help="Project name (default: current directory name)",
    )
    init_parser.add_argument(
        "--template",
        choices=["basic", "manim", "math", "lib"],
        default="basic",
        help="Project template (default: basic)",
    )
    init_parser.add_argument(
        "--no-git",
        action="store_true",
        help="Don't initialize git repository",
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        aliases=["b"],
        help="Build the MathViz project",
    )
    build_parser.add_argument(
        "--release",
        action="store_true",
        help="Build with optimizations enabled",
    )
    build_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed build information",
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        aliases=["t"],
        help="Run project tests",
    )
    test_parser.add_argument(
        "pattern",
        type=str,
        nargs="?",
        default=None,
        help="Test file pattern to match",
    )
    test_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose test output",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show language information and capabilities",
    )

    return parser


def cmd_compile(args: argparse.Namespace) -> int:
    """Handle the compile command."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")

        # Parse the source first
        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        # Type checking (unless disabled)
        if not args.no_typecheck:
            # Create type checker with source for rich diagnostics
            checker = TypeChecker(source, str(input_path))
            errors = checker.check(ast, source, str(input_path))

            if errors:
                error_count = len(errors)
                if args.strict or error_count > 0:
                    print(f"{Colors.RED}Type errors found:{Colors.RESET}", file=sys.stderr)

                    # Print rich diagnostics if available
                    diagnostics = checker.get_diagnostics()
                    if diagnostics:
                        for diagnostic in diagnostics:
                            print(diagnostic.render(source, use_color=sys.stdout.isatty()))
                            print()
                    else:
                        # Fall back to simple error messages
                        for error in errors:
                            _print_type_error(error)

                    if args.strict:
                        return 1

            if args.verbose:
                _print_verbose_type_info(checker, errors)

        # Verbose analysis information
        if args.verbose:
            # Purity analysis
            purity_info = analyze_purity(ast)
            print(f"\n{Colors.BOLD}Purity Analysis:{Colors.RESET}")
            for func_name, info in purity_info.items():
                purity_color = Colors.GREEN if info.purity == Purity.PURE else Colors.YELLOW
                print(f"  {func_name}: {purity_color}{info.purity.name}{Colors.RESET}")

            # Complexity analysis
            complexity_info = analyze_complexity(ast)
            print(f"\n{Colors.BOLD}Complexity Analysis:{Colors.RESET}")
            for func_name, info in complexity_info.items():
                print(f"  {func_name}: {Colors.CYAN}{info.complexity.value}{Colors.RESET}")

        # Generate code with optional parallelization analysis
        parallel_info = None
        if not args.no_parallel:
            # Run parallel analysis if parallelization is enabled
            from mathviz.compiler.ast_nodes import FunctionDef

            parallel_analyzer = ParallelAnalyzer()
            parallel_info = {}
            for stmt in ast.statements:
                if isinstance(stmt, FunctionDef):
                    loops = parallel_analyzer.analyze_function(stmt)
                    if loops:
                        # Convert to the format CodeGenerator expects
                        parallel_info[stmt.name] = [analysis for _, analysis in loops]

        generator = CodeGenerator(
            optimize=not args.no_optimize,
            parallel_info=parallel_info,
            verbose=args.verbose,
        )
        python_code = generator.generate(ast)

        if args.stdout:
            print(python_code)
        else:
            output_path = args.output or input_path.with_suffix(".py")
            output_path.write_text(python_code, encoding="utf-8")
            print(f"{Colors.GREEN}Compiled:{Colors.RESET} {input_path} -> {output_path}")

        return 0

    except MathVizError as e:
        print(f"{Colors.RED}Compilation error:{Colors.RESET} {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{Colors.RED}Internal error:{Colors.RESET} {e}", file=sys.stderr)
        return 1


def _print_type_error(error: MathVizTypeError) -> None:
    """Print a formatted type error with source location."""
    loc = error.location
    if loc:
        print(
            f"  {Colors.GRAY}{loc.filename or '<input>'}:{loc.line}:{loc.column}:{Colors.RESET} "
            f"{Colors.RED}error:{Colors.RESET} {error.message}"
        )
    else:
        print(f"  {Colors.RED}error:{Colors.RESET} {error.message}")


def _print_verbose_type_info(checker: TypeChecker, errors: list[MathVizTypeError]) -> None:
    """Print verbose type checking information."""
    error_count = len(errors)
    warning_count = 0  # TODO: Separate warnings from errors when we add warning support

    if error_count == 0:
        print(
            f"{Colors.GREEN}[check]{Colors.RESET} Type check passed ({error_count} errors, {warning_count} warnings)"
        )
    else:
        print(
            f"{Colors.YELLOW}[check]{Colors.RESET} Type check completed ({error_count} errors, {warning_count} warnings)"
        )


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the run command."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")
        python_code = compile_source(source, optimize=not args.no_optimize)

        # Write to temporary file
        temp_path = input_path.with_suffix(".py")
        temp_path.write_text(python_code, encoding="utf-8")

        # Build manim command
        quality_map = {
            "l": "-ql",
            "m": "-qm",
            "h": "-qh",
            "p": "-qp",
            "k": "-qk",
        }
        quality_flag = quality_map.get(args.quality, "-qm")

        # Use sys.executable to run manim with the same Python interpreter
        # This ensures manim is found when installed via pipx/uv
        cmd = [sys.executable, "-m", "manim", quality_flag]
        if args.preview:
            cmd.append("-p")
        cmd.append(str(temp_path))

        # Run manim
        result = subprocess.run(cmd)
        return result.returncode

    except MathVizError as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        return 1
    except ModuleNotFoundError:
        print("Error: manim not found. Install with: pip install manim", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Handle the check command."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    parser = None  # Initialize for exception handler
    source = ""
    try:
        source = input_path.read_text(encoding="utf-8")

        # Lexing
        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()

        # Parsing with rich diagnostics
        parser = Parser(tokens, source, str(input_path))
        parser.parse()

        print(f"{Colors.GREEN}OK:{Colors.RESET} {input_path} (no syntax errors)")
        return 0

    except MathVizError as e:
        # Try to get rich diagnostics from parser
        if parser is not None and hasattr(parser, "diagnostics") and parser.diagnostics:
            use_color = sys.stdout.isatty()
            for diagnostic in parser.diagnostics:
                print(diagnostic.render(source, use_color=use_color), file=sys.stderr)
                print(file=sys.stderr)
        else:
            print(f"{Colors.RED}Error:{Colors.RESET} {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{Colors.RED}Internal error:{Colors.RESET} {e}", file=sys.stderr)
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run comprehensive analysis on a MathViz file."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")

        # Parse the source
        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        # Run all analyzers
        type_checker = TypeChecker()
        type_errors = type_checker.check(ast)

        purity_info = analyze_purity(ast)
        complexity_info = analyze_complexity(ast)

        call_graph_builder = CallGraphBuilder()
        call_graph = call_graph_builder.build(ast)

        parallel_analyzer = ParallelAnalyzer()
        parallel_info: dict[str, list[tuple[Any, Any]]] = {}

        # Collect parallelizable loops for each function
        from mathviz.compiler.ast_nodes import FunctionDef

        for stmt in ast.statements:
            if isinstance(stmt, FunctionDef):
                loops = parallel_analyzer.analyze_function(stmt)
                if loops:
                    parallel_info[stmt.name] = loops

        if args.json:
            _print_analysis_json(
                input_path,
                type_errors,
                purity_info,
                complexity_info,
                call_graph,
                parallel_info,
            )
        else:
            _print_analysis_report(
                input_path,
                type_errors,
                purity_info,
                complexity_info,
                call_graph,
                parallel_info,
                type_checker,
            )

        return 0

    except MathVizError as e:
        print(f"{Colors.RED}Error:{Colors.RESET} {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{Colors.RED}Internal error:{Colors.RESET} {e}", file=sys.stderr)
        return 1


def _print_analysis_report(
    input_path: Path,
    type_errors: list[MathVizTypeError],
    purity_info: dict[str, Any],
    complexity_info: dict[str, Any],
    call_graph: CallGraph,
    parallel_info: dict[str, list[tuple[Any, Any]]],
    type_checker: TypeChecker,
) -> None:
    """Print formatted analysis report with colors."""
    print(f"\n{Colors.BOLD}Analysis Report: {input_path}{Colors.RESET}")
    print("=" * 60)

    # Type check results
    error_count = len(type_errors)
    warning_count = 0  # Future: separate warnings

    if error_count == 0:
        print(
            f"\n{Colors.GREEN}[ok]{Colors.RESET} Type check passed ({error_count} errors, {warning_count} warnings)"
        )
    else:
        print(f"\n{Colors.RED}[!!]{Colors.RESET} Type check found {error_count} error(s)")
        for error in type_errors:
            _print_type_error(error)

    # Functions analyzed section
    if purity_info or complexity_info:
        print(f"\n{Colors.BOLD}Functions analyzed:{Colors.RESET}")

        all_funcs = set(purity_info.keys()) | set(complexity_info.keys())

        for func_name in sorted(all_funcs):
            # Get function signature from type checker
            sig = type_checker.function_signatures.get(func_name)
            if sig:
                params = ", ".join(sig.param_names) if sig.param_names else ""
                print(f"\n  {Colors.CYAN}{func_name}{Colors.RESET}({params})")
            else:
                print(f"\n  {Colors.CYAN}{func_name}{Colors.RESET}()")

            # Purity
            if func_name in purity_info:
                pinfo = purity_info[func_name]
                purity = pinfo.purity
                if purity == Purity.PURE:
                    print(f"    Purity: {Colors.GREEN}PURE{Colors.RESET}")
                elif purity == Purity.IMPURE_IO:
                    print(f"    Purity: {Colors.YELLOW}IMPURE (I/O){Colors.RESET}")
                elif purity == Purity.IMPURE_MUTATION:
                    print(f"    Purity: {Colors.YELLOW}IMPURE (mutation){Colors.RESET}")
                elif purity == Purity.IMPURE_MANIM:
                    print(f"    Purity: {Colors.YELLOW}IMPURE (Manim){Colors.RESET}")

            # Complexity
            if func_name in complexity_info:
                cinfo = complexity_info[func_name]
                print(f"    Complexity: {Colors.MAGENTA}{cinfo.complexity.value}{Colors.RESET}")

            # Parallelizable loops
            if func_name in parallel_info:
                loops = parallel_info[func_name]
                parallelizable_count = sum(1 for _, analysis in loops if analysis.is_parallelizable)
                if parallelizable_count > 0:
                    print(
                        f"    Parallelizable loops: {Colors.GREEN}{parallelizable_count}{Colors.RESET}"
                    )

            # JIT compatibility
            if func_name in purity_info:
                pinfo = purity_info[func_name]
                if is_jit_safe(pinfo):
                    print(f"    JIT: {Colors.GREEN}[ok] Compatible{Colors.RESET}")
                else:
                    print(f"    JIT: {Colors.GRAY}[-] Not recommended{Colors.RESET}")

    # Call Graph section
    if call_graph.nodes:
        print(f"\n{Colors.BOLD}Call Graph:{Colors.RESET}")
        for node_name, node in sorted(call_graph.nodes.items()):
            if node.calls:
                callees = ", ".join(sorted(node.calls))
                print(f"  {node_name} -> {callees}")

    # Optimization suggestions
    suggestions = _get_optimization_suggestions(purity_info, complexity_info, parallel_info)
    if suggestions:
        print(f"\n{Colors.BOLD}Optimization suggestions:{Colors.RESET}")
        for suggestion in suggestions:
            print(f"  {Colors.YELLOW}*{Colors.RESET} {suggestion}")

    print()


def _get_optimization_suggestions(
    purity_info: dict[str, Any],
    complexity_info: dict[str, Any],
    parallel_info: dict[str, list[tuple[Any, Any]]],
) -> list[str]:
    """Generate optimization suggestions based on analysis."""
    suggestions: list[str] = []

    for func_name, cinfo in complexity_info.items():
        if cinfo.complexity in (Complexity.O_N_SQUARED, Complexity.O_N_CUBED, Complexity.O_2_N):
            suggestions.append(
                f"Function '{func_name}' has {cinfo.complexity.value} complexity - consider algorithmic optimization"
            )

    for func_name, loops in parallel_info.items():
        for loop, analysis in loops:
            if analysis.is_parallelizable and not analysis.can_use_prange:
                suggestions.append(
                    f"Loop in '{func_name}' could be parallelized with some refactoring"
                )

    for func_name, pinfo in purity_info.items():
        if pinfo.purity == Purity.PURE and func_name in complexity_info:
            cinfo = complexity_info[func_name]
            if cinfo.complexity != Complexity.O_1:
                suggestions.append(
                    f"Pure function '{func_name}' is a good candidate for memoization"
                )

    return suggestions


def _print_analysis_json(
    input_path: Path,
    type_errors: list[MathVizTypeError],
    purity_info: dict[str, Any],
    complexity_info: dict[str, Any],
    call_graph: CallGraph,
    parallel_info: dict[str, list[tuple[Any, Any]]],
) -> None:
    """Print analysis results as JSON."""
    result: dict[str, Any] = {
        "file": str(input_path),
        "type_check": {
            "errors": [
                {
                    "message": e.message,
                    "location": {
                        "line": e.location.line if e.location else None,
                        "column": e.location.column if e.location else None,
                    }
                    if e.location
                    else None,
                }
                for e in type_errors
            ],
            "passed": len(type_errors) == 0,
        },
        "functions": {},
        "call_graph": {},
    }

    # Build function analysis
    all_funcs = set(purity_info.keys()) | set(complexity_info.keys())
    for func_name in sorted(all_funcs):
        func_data: dict[str, Any] = {}

        if func_name in purity_info:
            pinfo = purity_info[func_name]
            func_data["purity"] = pinfo.purity.name
            func_data["is_pure"] = pinfo.is_pure()
            func_data["has_io"] = pinfo.has_io()
            func_data["has_manim_calls"] = pinfo.has_manim_calls()
            func_data["jit_safe"] = is_jit_safe(pinfo)

        if func_name in complexity_info:
            cinfo = complexity_info[func_name]
            func_data["complexity"] = cinfo.complexity.value
            func_data["loop_depth"] = cinfo.loop_depth
            func_data["has_recursion"] = cinfo.has_recursion

        if func_name in parallel_info:
            loops = parallel_info[func_name]
            func_data["parallelizable_loops"] = sum(
                1 for _, analysis in loops if analysis.is_parallelizable
            )

        result["functions"][func_name] = func_data

    # Build call graph
    for node_name, node in call_graph.nodes.items():
        result["call_graph"][node_name] = {
            "calls": list(sorted(node.calls)),
            "called_by": list(sorted(node.called_by)),
            "is_recursive": node.is_recursive,
            "in_cycle": node.in_cycle,
        }

    print(json.dumps(result, indent=2))


def cmd_typecheck(args: argparse.Namespace) -> int:
    """Type check a MathViz file."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")

        # Parse the source
        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        # Run type checker with rich diagnostics
        checker = TypeChecker(source, str(input_path))
        errors = checker.check(ast, source, str(input_path))

        # Count errors and warnings
        error_count = len(errors)
        warning_count = 0  # Future: separate warnings from errors

        if error_count == 0:
            print(
                f"{Colors.GREEN}[ok]{Colors.RESET} Type check passed ({error_count} errors, {warning_count} warnings)"
            )
            return 0
        else:
            print(
                f"{Colors.RED}[!!]{Colors.RESET} Type check failed ({error_count} errors, {warning_count} warnings)"
            )
            print()

            # Use rich diagnostics if available
            diagnostics = checker.get_diagnostics()
            use_color = sys.stdout.isatty()
            if diagnostics:
                for diagnostic in diagnostics:
                    print(diagnostic.render(source, use_color=use_color))
                    print()
            else:
                # Fall back to simple error messages
                for error in errors:
                    _print_type_error(error)
                print()

            if args.strict:
                return 1
            # Without strict, only return 1 if there are actual errors
            return 1 if error_count > 0 else 0

    except MathVizError as e:
        print(f"{Colors.RED}Error:{Colors.RESET} {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{Colors.RED}Internal error:{Colors.RESET} {e}", file=sys.stderr)
        return 1


def cmd_tokens(args: argparse.Namespace) -> int:
    """Handle the tokens command (debug)."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")
        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()

        for token in tokens:
            print(token)

        return 0

    except MathVizError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_ast(args: argparse.Namespace) -> int:
    """Handle the ast command (debug)."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")

        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse()

        # Pretty print the AST
        _print_ast(ast)

        return 0

    except MathVizError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_new(args: argparse.Namespace) -> int:
    """Handle the new command - create a new project."""
    name = args.name
    template = args.template

    project_dir = Path(name)
    if project_dir.exists():
        print(f"Error: Directory '{name}' already exists", file=sys.stderr)
        return 1

    project_dir.mkdir(parents=True)

    # Create main file based on template
    templates = {
        "basic": """# {name} - MathViz Project

fn main() {{
    println("Hello from {name}!")
}}
""",
        "manim": """# {name} - MathViz Animation Project
use manim.*

fn main() {{
    println("Run with: mathviz run {name}/main.mviz")
}}

scene MainScene extends Scene {{
    fn construct(self) {{
        let title = Text("{name}")
        play(Write(title))
        wait(2.0)
        play(FadeOut(title))
    }}
}}
""",
        "math": """# {name} - MathViz Math Project

mod math_utils {{
    fn factorial(n: Int) -> Int {{
        if n <= 1 {{ return 1 }}
        return n * factorial(n - 1)
    }}
}}

fn main() {{
    println("=== {name} ===")
    println("5! = {{}}", math_utils.factorial(5))

    let arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    println("sum = {{}}", sum(arr))
}}
""",
    }

    main_content = templates[template].format(name=name)
    main_file = project_dir / "main.mviz"
    main_file.write_text(main_content, encoding="utf-8")

    print(f"Created new MathViz project: {name}/")
    print(f"  - main.mviz (template: {template})")
    print(f"\nTo compile: mathviz compile {name}/main.mviz")
    print(f"To run:     mathviz exec {name}/main.mviz")

    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    """Handle the exec command - compile and run without Manim."""
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")
        python_code = compile_source(source, optimize=not args.no_optimize)

        # Execute the generated code directly using Python's exec
        # This is safe because we're executing compiler-generated code
        # Set __name__ to "__main__" so that `if __name__ == "__main__":` block runs
        exec_globals: dict = {"__name__": "__main__"}
        compiled_code = compile(python_code, str(input_path), "exec")
        eval(compiled_code, exec_globals)  # noqa: S307

        return 0

    except MathVizError as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return 1


def cmd_lint(args: argparse.Namespace) -> int:
    """Handle the lint command."""
    # Handle --list-rules flag
    if args.list_rules:
        _print_lint_rules()
        return 0

    # Check if input was provided
    if args.input is None:
        print("Error: Input file is required (or use --list-rules)", file=sys.stderr)
        return 1

    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    try:
        source = input_path.read_text(encoding="utf-8")

        # Parse the source
        lexer = Lexer(source, str(input_path))
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        # Configure the linter
        config = LintConfiguration()

        if args.warn_all:
            config.warn_all()

        if args.deny:
            try:
                category = LintCategory(args.deny.lower())
                config.set_level_by_category(category, LintLevel.DENY)
            except ValueError:
                print(f"Error: Unknown category '{args.deny}'", file=sys.stderr)
                print(
                    "Valid categories: unused, unreachable, style, math, performance, correctness"
                )
                return 1

        if args.allow:
            rule_id = args.allow
            if rule_id in ALL_RULES:
                config.allow(rule_id)
            elif rule_id in RULES_BY_NAME:
                config.allow(rule_id)
            else:
                print(f"Error: Unknown rule '{rule_id}'", file=sys.stderr)
                return 1

        # Run the linter
        linter = Linter(config)
        violations = linter.lint(ast)

        # Output results
        if args.json:
            _print_lint_json(input_path, violations)
        else:
            _print_lint_report(input_path, violations, config, source)

        # Return code based on violations
        if args.fix:
            print(f"{Colors.YELLOW}Note:{Colors.RESET} Auto-fix is not yet implemented")

        # Count errors (deny level)
        error_count = sum(1 for v in violations if config.get_level(v.rule) == LintLevel.DENY)
        if error_count > 0:
            return 1

        return 0

    except MathVizError as e:
        print(f"{Colors.RED}Error:{Colors.RESET} {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{Colors.RED}Internal error:{Colors.RESET} {e}", file=sys.stderr)
        return 1


def _print_lint_report(
    input_path: Path,
    violations: list[LintViolation],
    config: LintConfiguration,
    source: Optional[str] = None,
) -> None:
    """
    Print a Rust-style formatted lint report with source context.

    Example output:
        warning[W0001]: variable 'x' is never used
          --> example.mviz:5:9
           |
         5 |     let x = 42
           |         ^ unused variable
           |
           = help: prefix with `_` to silence this warning
    """
    if not violations:
        print(f"{Colors.GREEN}[ok]{Colors.RESET} {input_path}: No lint issues found")
        return

    source_lines = source.splitlines() if source else []

    # Group by level
    errors = [v for v in violations if config.get_level(v.rule) == LintLevel.DENY]
    warnings = [v for v in violations if config.get_level(v.rule) == LintLevel.WARN]

    for violation in violations:
        level = config.get_level(violation.rule)
        level_color = Colors.RED if level == LintLevel.DENY else Colors.YELLOW
        level_str = "error" if level == LintLevel.DENY else "warning"

        # Header: warning[W0001]: variable 'x' is never used
        print(
            f"{level_color}{Colors.BOLD}{level_str}[{violation.rule.code}]{Colors.RESET}: "
            f"{Colors.BOLD}{violation.message}{Colors.RESET}"
        )

        # Location: --> example.mviz:5:9
        loc = violation.location
        if loc:
            print(f"  {Colors.BLUE}-->{Colors.RESET} {input_path}:{loc.line}:{loc.column}")

            # Source context with underline
            if source_lines and 1 <= loc.line <= len(source_lines):
                print(f"   {Colors.BLUE}|{Colors.RESET}")

                # Show the source line
                source_line = source_lines[loc.line - 1]
                line_num_str = f"{loc.line:3}"
                print(f"{Colors.BLUE}{line_num_str} |{Colors.RESET} {source_line}")

                # Underline the problematic span
                # Estimate the length based on the violation message
                underline_length = _estimate_span_length(violation)
                padding = " " * (loc.column - 1)
                underline = "^" * underline_length
                print(
                    f"   {Colors.BLUE}|{Colors.RESET} {padding}{level_color}{underline}{Colors.RESET}"
                )

                print(f"   {Colors.BLUE}|{Colors.RESET}")
        else:
            print(f"  {Colors.BLUE}-->{Colors.RESET} {input_path}:<unknown>")
            print(f"   {Colors.BLUE}|{Colors.RESET}")

        # Help/suggestion
        if violation.suggestion:
            print(
                f"   {Colors.BLUE}={Colors.RESET} {Colors.GREEN}help:{Colors.RESET} {violation.suggestion}"
            )

        # Related locations
        for related_loc in violation.related_locations:
            print(
                f"   {Colors.BLUE}={Colors.RESET} {Colors.BOLD}note:{Colors.RESET} related location at {related_loc}"
            )

        print()

    # Summary
    print(f"Found {len(errors)} error(s) and {len(warnings)} warning(s)")


def _estimate_span_length(violation: LintViolation) -> int:
    """Estimate the length of the problematic span for underlining."""
    # Try to extract the variable/function name from the message for accurate underlining
    import re

    # Match patterns like "'x'" or "`x`" in the message
    match = re.search(r"['\"`]([^'\"`]+)['\"`]", violation.message)
    if match:
        return len(match.group(1))

    # Default to a reasonable length
    return 1


def _print_lint_json(input_path: Path, violations: list[LintViolation]) -> None:
    """Print lint results as JSON."""
    result = {
        "file": str(input_path),
        "violations": [
            {
                "rule": {
                    "code": v.rule.code,
                    "name": v.rule.name,
                    "category": v.rule.category.value,
                },
                "location": {
                    "line": v.location.line if v.location else None,
                    "column": v.location.column if v.location else None,
                },
                "message": v.message,
                "suggestion": v.suggestion,
            }
            for v in violations
        ],
        "summary": {
            "total": len(violations),
            "by_category": {},
        },
    }

    # Count by category
    for v in violations:
        cat = v.rule.category.value
        result["summary"]["by_category"][cat] = result["summary"]["by_category"].get(cat, 0) + 1

    print(json.dumps(result, indent=2))


def _print_lint_rules() -> None:
    """Print all available lint rules."""
    print(f"\n{Colors.BOLD}Available Lint Rules{Colors.RESET}")
    print("=" * 60)

    # Group by category
    by_category: dict[LintCategory, list] = {}
    for rule in ALL_RULES.values():
        if rule.category not in by_category:
            by_category[rule.category] = []
        by_category[rule.category].append(rule)

    for category in LintCategory:
        if category not in by_category:
            continue

        print(f"\n{Colors.CYAN}{category.value.upper()}{Colors.RESET}")
        for rule in sorted(by_category[category], key=lambda r: r.code):
            level_str = (
                f"{Colors.YELLOW}warn{Colors.RESET}"
                if rule.level == LintLevel.WARN
                else f"{Colors.RED}deny{Colors.RESET}"
            )
            print(f"  {rule.code} {rule.name:30s} [{level_str}]")
            # Truncate message for display
            msg = rule.message.replace("{}", "<name>")
            if len(msg) > 50:
                msg = msg[:47] + "..."
            print(f"    {Colors.GRAY}{msg}{Colors.RESET}")

    print()


def cmd_repl(args: argparse.Namespace) -> int:
    """Handle the repl command - start interactive mode."""
    from mathviz.repl import REPLSession

    session = REPLSession()
    session.run()
    return 0


def cmd_fmt(args: argparse.Namespace) -> int:
    """Handle the fmt command - format source files."""
    from mathviz.formatter import format_file, format_source, check_format, get_diff

    # Default to current directory if no input specified
    input_path = args.input or Path(".")

    # Collect all .mviz files
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.rglob("*.mviz"))
        if not files:
            print(f"No .mviz files found in {input_path}", file=sys.stderr)
            return 0
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1

    exit_code = 0

    for filepath in files:
        try:
            source = filepath.read_text(encoding="utf-8")

            if args.check:
                # Check mode: just report if formatting is needed
                is_formatted = check_format(source)
                if not is_formatted:
                    print(f"{Colors.YELLOW}Would reformat:{Colors.RESET} {filepath}")
                    exit_code = 1
            elif args.diff:
                # Diff mode: show what would change
                diff = get_diff(source, filename=str(filepath))
                if diff:
                    print(diff)
                    exit_code = 1
            elif args.write:
                # Write mode: format in place
                formatted = format_source(source)
                if source != formatted:
                    filepath.write_text(formatted, encoding="utf-8")
                    print(f"{Colors.GREEN}Formatted:{Colors.RESET} {filepath}")
            else:
                # Default: print formatted output to stdout
                formatted = format_source(source)
                print(formatted)

        except Exception as e:
            print(f"{Colors.RED}Error formatting {filepath}:{Colors.RESET} {e}", file=sys.stderr)
            exit_code = 1

    if args.check and exit_code == 0:
        print(f"{Colors.GREEN}All files are properly formatted{Colors.RESET}")

    return exit_code


def cmd_watch(args: argparse.Namespace) -> int:
    """Handle the watch command - watch and recompile on changes."""
    import time

    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1

    # Determine output path
    if input_path.is_file():
        output_path = args.output or input_path.with_suffix(".py")
        watch_files = [input_path]
    else:
        output_path = args.output or input_path
        watch_files = list(input_path.rglob("*.mviz"))

    print(f"{Colors.BOLD}Watching for changes...{Colors.RESET}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"Press Ctrl+C to stop\n")

    # Track file modification times
    last_mtime: dict[Path, float] = {}
    for f in watch_files:
        if f.exists():
            last_mtime[f] = f.stat().st_mtime

    def compile_all() -> None:
        """Compile all watched files."""
        for filepath in watch_files:
            try:
                source = filepath.read_text(encoding="utf-8")
                python_code = compile_source(source, optimize=not args.no_optimize)
                out_file = (
                    output_path
                    if output_path.suffix == ".py"
                    else output_path / filepath.with_suffix(".py").name
                )
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(python_code, encoding="utf-8")
                print(
                    f"{Colors.GREEN}[{time.strftime('%H:%M:%S')}] Compiled:{Colors.RESET} {filepath}"
                )
            except MathVizError as e:
                print(f"{Colors.RED}[{time.strftime('%H:%M:%S')}] Error:{Colors.RESET} {e}")
            except Exception as e:
                print(f"{Colors.RED}[{time.strftime('%H:%M:%S')}] Error:{Colors.RESET} {e}")

    # Initial compile
    compile_all()

    try:
        while True:
            time.sleep(0.5)

            # Check for changes
            changed = False
            for filepath in watch_files:
                if filepath.exists():
                    mtime = filepath.stat().st_mtime
                    if filepath not in last_mtime or mtime > last_mtime[filepath]:
                        last_mtime[filepath] = mtime
                        changed = True

            if changed:
                compile_all()

    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}Watch stopped{Colors.RESET}")
        return 0


def cmd_doc(args: argparse.Namespace) -> int:
    """Handle the doc command - generate documentation."""
    input_path = args.input or Path(".")
    output_path = args.output or Path("docs")

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1

    # Collect .mviz files
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob("*.mviz"))

    if not files:
        print(f"No .mviz files found in {input_path}", file=sys.stderr)
        return 0

    docs: list[dict[str, Any]] = []

    for filepath in files:
        try:
            source = filepath.read_text(encoding="utf-8")
            lexer = Lexer(source, str(filepath))
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()

            # Extract documentation from the AST
            file_doc = _extract_documentation(ast, filepath)
            docs.append(file_doc)

        except Exception as e:
            print(f"{Colors.YELLOW}Warning:{Colors.RESET} Could not process {filepath}: {e}")

    if args.json:
        print(json.dumps(docs, indent=2))
    elif args.html:
        # Generate HTML documentation
        output_path.mkdir(parents=True, exist_ok=True)
        _generate_html_docs(docs, output_path)
        print(f"{Colors.GREEN}Generated HTML documentation in {output_path}{Colors.RESET}")
    else:
        # Print plain text documentation
        for doc in docs:
            _print_doc(doc)

    return 0


def _extract_documentation(ast, filepath: Path) -> dict[str, Any]:
    """Extract documentation from an AST."""
    from mathviz.compiler.ast_nodes import FunctionDef, StructDef, TraitDef, EnumDef, SceneDef

    doc: dict[str, Any] = {
        "file": str(filepath),
        "functions": [],
        "structs": [],
        "traits": [],
        "enums": [],
        "scenes": [],
    }

    for stmt in ast.statements:
        if isinstance(stmt, FunctionDef):
            func_doc = {
                "name": stmt.name,
                "params": [
                    {
                        "name": p.name,
                        "type": str(p.type_annotation.name)
                        if p.type_annotation and hasattr(p.type_annotation, "name")
                        else None,
                    }
                    for p in stmt.parameters
                ],
                "return_type": str(stmt.return_type.name)
                if stmt.return_type and hasattr(stmt.return_type, "name")
                else None,
                "jit": stmt.jit_options.mode.value if stmt.jit_options else None,
            }
            doc["functions"].append(func_doc)
        elif isinstance(stmt, StructDef):
            doc["structs"].append(
                {
                    "name": stmt.name,
                    "fields": [
                        {"name": f.name, "type": str(f.type_annotation)} for f in stmt.fields
                    ],
                }
            )
        elif isinstance(stmt, TraitDef):
            doc["traits"].append(
                {
                    "name": stmt.name,
                    "methods": [m.name for m in stmt.methods],
                }
            )
        elif isinstance(stmt, EnumDef):
            doc["enums"].append(
                {
                    "name": stmt.name,
                    "variants": [v.name for v in stmt.variants],
                }
            )
        elif isinstance(stmt, SceneDef):
            doc["scenes"].append({"name": stmt.name})

    return doc


def _print_doc(doc: dict[str, Any]) -> None:
    """Print documentation to terminal."""
    print(f"\n{Colors.BOLD}=== {doc['file']} ==={Colors.RESET}\n")

    if doc["functions"]:
        print(f"{Colors.CYAN}Functions:{Colors.RESET}")
        for func in doc["functions"]:
            params = ", ".join(
                f"{p['name']}: {p['type']}" if p["type"] else p["name"] for p in func["params"]
            )
            ret = f" -> {func['return_type']}" if func["return_type"] else ""
            jit = (
                f" {Colors.GREEN}@{func['jit']}{Colors.RESET}"
                if func["jit"] and func["jit"] != "none"
                else ""
            )
            print(f"  {func['name']}({params}){ret}{jit}")

    if doc["structs"]:
        print(f"\n{Colors.CYAN}Structs:{Colors.RESET}")
        for struct in doc["structs"]:
            print(f"  {struct['name']}")
            for field in struct["fields"]:
                print(f"    {field['name']}: {field['type']}")

    if doc["traits"]:
        print(f"\n{Colors.CYAN}Traits:{Colors.RESET}")
        for trait in doc["traits"]:
            print(f"  {trait['name']}: {', '.join(trait['methods'])}")

    if doc["enums"]:
        print(f"\n{Colors.CYAN}Enums:{Colors.RESET}")
        for enum in doc["enums"]:
            print(f"  {enum['name']}: {', '.join(enum['variants'])}")

    if doc["scenes"]:
        print(f"\n{Colors.CYAN}Scenes:{Colors.RESET}")
        for scene in doc["scenes"]:
            print(f"  {scene['name']}")


def _generate_html_docs(docs: list[dict[str, Any]], output_path: Path) -> None:
    """Generate HTML documentation."""
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>MathViz Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #1e1e1e; color: #d4d4d4; }}
        h1 {{ color: #569cd6; }}
        h2 {{ color: #4ec9b0; border-bottom: 1px solid #3e3e3e; padding-bottom: 5px; }}
        h3 {{ color: #dcdcaa; }}
        .function {{ background: #252526; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .signature {{ font-family: 'Consolas', monospace; color: #ce9178; }}
        .type {{ color: #4ec9b0; }}
        .jit {{ color: #6a9955; font-size: 0.8em; }}
    </style>
</head>
<body>
    <h1>MathViz Documentation</h1>
    {content}
</body>
</html>"""

    content_parts = []
    for doc in docs:
        content_parts.append(f"<h2>{doc['file']}</h2>")

        if doc["functions"]:
            content_parts.append("<h3>Functions</h3>")
            for func in doc["functions"]:
                params = ", ".join(
                    f'{p["name"]}: <span class="type">{p["type"]}</span>'
                    if p["type"]
                    else p["name"]
                    for p in func["params"]
                )
                ret = (
                    f' -&gt; <span class="type">{func["return_type"]}</span>'
                    if func["return_type"]
                    else ""
                )
                jit = (
                    f' <span class="jit">@{func["jit"]}</span>'
                    if func["jit"] and func["jit"] != "none"
                    else ""
                )
                content_parts.append(
                    f'<div class="function"><span class="signature">{func["name"]}({params}){ret}</span>{jit}</div>'
                )

        if doc["structs"]:
            content_parts.append("<h3>Structs</h3>")
            for struct in doc["structs"]:
                fields = ", ".join(f"{f['name']}: {f['type']}" for f in struct["fields"])
                content_parts.append(
                    f'<div class="function"><span class="signature">{struct["name"]} {{ {fields} }}</span></div>'
                )

    html = html_template.format(content="\n".join(content_parts))
    (output_path / "index.html").write_text(html, encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> int:
    """Handle the init command - initialize a new project."""
    import os

    name = args.name
    template = args.template

    # If no name provided, use current directory
    if name:
        project_dir = Path(name)
        if project_dir.exists() and any(project_dir.iterdir()):
            print(f"Error: Directory '{name}' is not empty", file=sys.stderr)
            return 1
        project_dir.mkdir(parents=True, exist_ok=True)
    else:
        project_dir = Path.cwd()
        name = project_dir.name

    # Create project structure
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)

    # Create main file based on template
    templates = {
        "basic": """# {name} - MathViz Project
# Created with `mathviz init`

fn main() {{
    println("Hello from {name}!")
}}
""",
        "manim": """# {name} - MathViz Animation Project
# Created with `mathviz init --template=manim`

use manim.*

scene MainScene {{
    fn construct(self) {{
        let title = Text("{name}")
        play(Write(title))
        wait(2.0)
        play(FadeOut(title))
    }}
}}

fn main() {{
    println("Run with: mathviz run src/main.mviz")
}}
""",
        "math": """# {name} - MathViz Math Project
# Created with `mathviz init --template=math`

@njit
fn factorial(n: Int) -> Int {{
    if n <= 1 {{ return 1 }}
    return n * factorial(n - 1)
}}

@njit(parallel=true)
fn dot_product(a: List[Float], b: List[Float]) -> Float {{
    let result = 0.0
    for i in 0..len(a) {{
        result += a[i] * b[i]
    }}
    return result
}}

fn main() {{
    println("=== {name} ===")
    println("5! = {{}}", factorial(5))

    let a = [1.0, 2.0, 3.0]
    let b = [4.0, 5.0, 6.0]
    println("dot product = {{}}", dot_product(a, b))
}}
""",
        "lib": """# {name} - MathViz Library
# Created with `mathviz init --template=lib`

/// A simple math utilities library

pub mod utils {{
    /// Clamp a value between min and max
    pub fn clamp(value: Float, min_val: Float, max_val: Float) -> Float {{
        if value < min_val {{ return min_val }}
        if value > max_val {{ return max_val }}
        return value
    }}

    /// Linear interpolation between two values
    pub fn lerp(a: Float, b: Float, t: Float) -> Float {{
        return a + (b - a) * t
    }}
}}
""",
    }

    main_content = templates[template].format(name=name)
    main_file = project_dir / "src" / "main.mviz"
    main_file.write_text(main_content, encoding="utf-8")

    # Create mathviz.toml configuration
    config_content = f'''# MathViz Project Configuration

[project]
name = "{name}"
version = "0.1.0"
template = "{template}"

[build]
optimize = true
parallel = true

[lint]
warn_all = false
'''
    (project_dir / "mathviz.toml").write_text(config_content, encoding="utf-8")

    # Create .gitignore
    gitignore_content = """# MathViz generated files
*.py
!tests/*.py

# Python
__pycache__/
*.pyc
.venv/

# IDE
.vscode/
.idea/

# Build
build/
dist/
docs/

# Media
media/
"""
    (project_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    # Initialize git if not disabled
    if not args.no_git:
        os.system(f"cd {project_dir} && git init -q 2>/dev/null || true")

    print(f"{Colors.GREEN}Created MathViz project:{Colors.RESET} {project_dir}/")
    print(f"  - src/main.mviz (template: {template})")
    print(f"  - mathviz.toml")
    print(f"  - .gitignore")
    print()
    print(f"Next steps:")
    print(f"  cd {project_dir}")
    print(f"  mathviz build     # Build the project")
    print(f"  mathviz exec src/main.mviz  # Run")

    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Handle the build command - build the project."""
    # Look for mathviz.toml in current directory
    config_file = Path("mathviz.toml")
    if not config_file.exists():
        print(
            f"{Colors.YELLOW}Warning:{Colors.RESET} No mathviz.toml found. Looking for .mviz files..."
        )

    # Find source files
    src_dir = Path("src")
    if src_dir.exists():
        files = list(src_dir.rglob("*.mviz"))
    else:
        files = list(Path(".").rglob("*.mviz"))

    if not files:
        print(f"Error: No .mviz files found", file=sys.stderr)
        return 1

    # Create build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    success_count = 0
    error_count = 0

    for filepath in files:
        try:
            source = filepath.read_text(encoding="utf-8")

            # Compile with optimizations if release build
            optimize = args.release

            python_code = compile_source(source, optimize=optimize)

            # Write output
            output_file = build_dir / filepath.with_suffix(".py").name
            output_file.write_text(python_code, encoding="utf-8")

            if args.verbose:
                print(f"  {Colors.GREEN}Compiled:{Colors.RESET} {filepath} -> {output_file}")

            success_count += 1

        except MathVizError as e:
            print(f"  {Colors.RED}Error:{Colors.RESET} {filepath}: {e}")
            error_count += 1
        except Exception as e:
            print(f"  {Colors.RED}Error:{Colors.RESET} {filepath}: {e}")
            error_count += 1

    # Summary
    if error_count == 0:
        print(f"{Colors.GREEN}Build successful:{Colors.RESET} {success_count} file(s) compiled")
    else:
        print(
            f"{Colors.YELLOW}Build completed with errors:{Colors.RESET} {success_count} succeeded, {error_count} failed"
        )

    return 1 if error_count > 0 else 0


def cmd_test(args: argparse.Namespace) -> int:
    """Handle the test command - run project tests."""
    tests_dir = Path("tests")

    if not tests_dir.exists():
        print(f"Error: No tests directory found", file=sys.stderr)
        return 1

    # Find test files
    pattern = args.pattern or "test_*.mviz"
    test_files = list(tests_dir.glob(pattern))

    if not test_files:
        print(f"No test files matching '{pattern}' found in tests/")
        return 0

    print(f"{Colors.BOLD}Running tests...{Colors.RESET}\n")

    passed = 0
    failed = 0

    for test_file in test_files:
        try:
            source = test_file.read_text(encoding="utf-8")
            python_code = compile_source(source, optimize=False)

            # Execute the test - using compile() + eval() for compiler-generated code
            # This is safe because we're only executing code generated by our compiler
            exec_globals: dict = {"__test_passed": True}
            compiled_code = compile(python_code, str(test_file), "exec")
            __builtins__["eval"](compiled_code, exec_globals)  # noqa: S307

            print(f"  {Colors.GREEN}PASS{Colors.RESET} {test_file.name}")
            passed += 1

        except Exception as e:
            print(f"  {Colors.RED}FAIL{Colors.RESET} {test_file.name}")
            if args.verbose:
                print(f"       {Colors.GRAY}{e}{Colors.RESET}")
            failed += 1

    print()
    if failed == 0:
        print(f"{Colors.GREEN}All tests passed:{Colors.RESET} {passed} test(s)")
        return 0
    else:
        print(f"{Colors.RED}Tests failed:{Colors.RESET} {passed} passed, {failed} failed")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info command - show language information."""
    print(f"""
{Colors.BOLD}MathViz Language{Colors.RESET}
================

{Colors.CYAN}Version:{Colors.RESET} {__version__}

{Colors.CYAN}Features:{Colors.RESET}
  - Static typing with type inference
  - Pattern matching with guards
  - Generics and traits
  - Algebraic data types (enums, structs)
  - Automatic JIT compilation via Numba
  - Auto-parallelization of loops
  - Manim integration for animations
  - Result and Optional types

{Colors.CYAN}Type System:{Colors.RESET}
  - Primitives: Int, Float, Bool, String, None
  - Collections: List[T], Set[T], Dict[K,V]
  - Mathematical: Vec, Mat, Array
  - Function types: (T1, T2) -> R
  - Optional[T] and Result[T, E]

{Colors.CYAN}JIT Decorators:{Colors.RESET}
  @jit           Standard JIT compilation
  @njit          Nopython JIT (faster, stricter)
  @vectorize     Create NumPy ufuncs
  @stencil       Stencil computations

{Colors.CYAN}Commands:{Colors.RESET}
  mathviz compile <file>     Compile to Python
  mathviz run <file>         Compile and run with Manim
  mathviz exec <file>        Compile and execute
  mathviz repl               Start interactive REPL
  mathviz fmt <file>         Format source code
  mathviz lint <file>        Run linter
  mathviz typecheck <file>   Type check
  mathviz analyze <file>     Comprehensive analysis
  mathviz watch <file>       Watch and recompile
  mathviz doc <dir>          Generate documentation
  mathviz init <name>        Create new project
  mathviz build              Build project
  mathviz test               Run tests

{Colors.CYAN}Documentation:{Colors.RESET}
  https://github.com/CyberSnakeH/MathViz
""")
    return 0


def _print_ast(node, indent: int = 0) -> None:
    """Pretty print an AST node."""
    prefix = "  " * indent
    node_name = type(node).__name__

    # Get relevant attributes
    attrs = {}
    for key in dir(node):
        if not key.startswith("_") and key not in ("accept", "location"):
            value = getattr(node, key)
            if not callable(value):
                attrs[key] = value

    # Print node
    if attrs:
        print(f"{prefix}{node_name}:")
        for key, value in attrs.items():
            if hasattr(value, "accept"):  # It's an AST node
                print(f"{prefix}  {key}:")
                _print_ast(value, indent + 2)
            elif isinstance(value, (list, tuple)) and value and hasattr(value[0], "accept"):
                print(f"{prefix}  {key}: [")
                for item in value:
                    _print_ast(item, indent + 2)
                print(f"{prefix}  ]")
            else:
                print(f"{prefix}  {key}: {value!r}")
    else:
        print(f"{prefix}{node_name}")


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    command_handlers = {
        "compile": cmd_compile,
        "c": cmd_compile,
        "run": cmd_run,
        "r": cmd_run,
        "check": cmd_check,
        "analyze": cmd_analyze,
        "a": cmd_analyze,
        "typecheck": cmd_typecheck,
        "tc": cmd_typecheck,
        "tokens": cmd_tokens,
        "ast": cmd_ast,
        "new": cmd_new,
        "exec": cmd_exec,
        "e": cmd_exec,
        "lint": cmd_lint,
        "l": cmd_lint,
        # New commands
        "repl": cmd_repl,
        "i": cmd_repl,
        "fmt": cmd_fmt,
        "format": cmd_fmt,
        "watch": cmd_watch,
        "w": cmd_watch,
        "doc": cmd_doc,
        "init": cmd_init,
        "build": cmd_build,
        "b": cmd_build,
        "test": cmd_test,
        "t": cmd_test,
        "info": cmd_info,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
