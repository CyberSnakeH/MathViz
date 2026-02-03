"""
Pytest configuration and shared fixtures for MathViz tests.
"""

from dataclasses import dataclass, field

import pytest

from mathviz.compiler.ast_nodes import Program
from mathviz.compiler.call_graph import CallGraph, CallGraphBuilder
from mathviz.compiler.codegen import CodeGenerator
from mathviz.compiler.complexity_analyzer import (
    ComplexityInfo,
    analyze_complexity,
)
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parallel_analyzer import (
    LoopAnalysis,
    ParallelAnalyzer,
)
from mathviz.compiler.parser import Parser
from mathviz.compiler.purity_analyzer import PurityInfo, analyze_purity
from mathviz.compiler.tokens import Token
from mathviz.compiler.type_checker import TypeChecker
from mathviz.utils.errors import TypeError as MathVizTypeError


@pytest.fixture
def lexer_factory():
    """Factory fixture for creating lexers."""

    def _create_lexer(source: str, filename: str = "test.mvz") -> Lexer:
        return Lexer(source, filename)

    return _create_lexer


@pytest.fixture
def parser_factory(lexer_factory):
    """Factory fixture for creating parsers from source."""

    def _create_parser(source: str) -> Parser:
        lexer = lexer_factory(source)
        tokens = lexer.tokenize()
        return Parser(tokens)

    return _create_parser


@pytest.fixture
def codegen_factory():
    """Factory fixture for creating code generators."""

    def _create_codegen(optimize: bool = False) -> CodeGenerator:
        return CodeGenerator(optimize=optimize)

    return _create_codegen


@pytest.fixture
def compile_source(parser_factory, codegen_factory):
    """Fixture to compile MathViz source to Python."""

    def _compile(source: str, optimize: bool = False) -> str:
        parser = parser_factory(source)
        ast = parser.parse()
        codegen = codegen_factory(optimize)
        return codegen.generate(ast)

    return _compile


@pytest.fixture
def tokenize(lexer_factory):
    """Fixture to tokenize source code."""

    def _tokenize(source: str) -> list[Token]:
        lexer = lexer_factory(source)
        return lexer.tokenize()

    return _tokenize


@pytest.fixture
def parse(parser_factory):
    """Fixture to parse source code into AST."""

    def _parse(source: str) -> Program:
        parser = parser_factory(source)
        return parser.parse()

    return _parse


@pytest.fixture
def compile_to_python(parser_factory, codegen_factory):
    """Fixture to compile MathViz source to Python code."""

    def _compile(source: str, optimize: bool = False) -> str:
        parser = parser_factory(source)
        ast = parser.parse()
        codegen = codegen_factory(optimize)
        return codegen.generate(ast)

    return _compile


# =============================================================================
# New Fixtures for Analysis Pipeline
# =============================================================================


@dataclass
class CompilationResult:
    """
    Complete result of compiling and analyzing a MathViz program.

    This dataclass holds all analysis results from the compilation pipeline,
    providing a unified view of type checking, purity analysis, complexity
    analysis, call graph construction, and parallelization detection.
    """

    # Source and AST
    source: str
    ast: Program

    # Generated code
    python_code: str
    optimized_code: str

    # Type checking
    type_errors: list[MathVizTypeError] = field(default_factory=list)
    type_checker: TypeChecker | None = None

    # Purity analysis
    purity_info: dict[str, PurityInfo] = field(default_factory=dict)

    # Complexity analysis
    complexity_info: dict[str, ComplexityInfo] = field(default_factory=dict)

    # Call graph
    call_graph: CallGraph | None = None

    # Parallelization analysis
    parallel_loops: list[tuple[str, LoopAnalysis]] = field(default_factory=list)

    def has_type_errors(self) -> bool:
        """Check if any type errors were detected."""
        return len(self.type_errors) > 0

    def get_pure_functions(self) -> list[str]:
        """Get list of function names that are pure."""
        return [name for name, info in self.purity_info.items() if info.is_pure()]

    def get_impure_functions(self) -> list[str]:
        """Get list of function names that are impure."""
        return [name for name, info in self.purity_info.items() if not info.is_pure()]

    def get_jit_candidates(self) -> list[str]:
        """Get list of functions that are candidates for JIT optimization."""
        from mathviz.compiler.purity_analyzer import is_jit_safe

        return [name for name, info in self.purity_info.items() if is_jit_safe(info)]

    def get_parallelizable_functions(self) -> list[str]:
        """Get list of functions containing parallelizable loops."""
        return list(
            {func_name for func_name, analysis in self.parallel_loops if analysis.is_parallelizable}
        )

    def get_recursive_functions(self) -> list[str]:
        """Get list of recursive functions from call graph."""
        if not self.call_graph:
            return []
        return [
            name
            for name, node in self.call_graph.nodes.items()
            if node.is_recursive or node.in_cycle
        ]

    def get_compilation_order(self) -> list[str]:
        """Get topological order of functions for compilation."""
        if not self.call_graph:
            return []
        try:
            return self.call_graph.topological_sort()
        except Exception:
            return []


@pytest.fixture
def compile_with_analysis(parse):
    """
    Fixture that returns full compilation result with all analyses.

    This fixture runs the complete compilation pipeline including:
    - Parsing
    - Type checking
    - Purity analysis
    - Complexity analysis
    - Call graph construction
    - Parallelization detection
    - Code generation (both optimized and non-optimized)
    """

    def _compile(source: str, optimize: bool = True, check_types: bool = True) -> CompilationResult:
        # Parse source to AST
        ast = parse(source)

        # Run type checking
        type_checker = TypeChecker()
        type_errors = type_checker.check(ast) if check_types else []

        # Run purity analysis
        purity_info = analyze_purity(ast)

        # Run complexity analysis
        complexity_info = analyze_complexity(ast)

        # Build call graph
        builder = CallGraphBuilder()
        call_graph = builder.build(ast)

        # Analyze parallelization for all functions
        parallel_analyzer = ParallelAnalyzer()
        parallel_loops: list[tuple[str, LoopAnalysis]] = []

        for stmt in ast.statements:
            from mathviz.compiler.ast_nodes import FunctionDef

            if isinstance(stmt, FunctionDef):
                loop_analyses = parallel_analyzer.analyze_function(stmt)
                for _loop, analysis in loop_analyses:
                    parallel_loops.append((stmt.name, analysis))

        # Generate code (non-optimized)
        codegen = CodeGenerator(optimize=False)
        python_code = codegen.generate(ast)

        # Generate optimized code
        optimized_codegen = CodeGenerator(optimize=optimize)
        optimized_code = optimized_codegen.generate(ast)

        return CompilationResult(
            source=source,
            ast=ast,
            python_code=python_code,
            optimized_code=optimized_code,
            type_errors=type_errors,
            type_checker=type_checker,
            purity_info=purity_info,
            complexity_info=complexity_info,
            call_graph=call_graph,
            parallel_loops=parallel_loops,
        )

    return _compile


@pytest.fixture
def analyze_types(parse):
    """Fixture to run type checking on source code."""

    def _analyze(source: str) -> tuple[Program, list[MathVizTypeError], TypeChecker]:
        ast = parse(source)
        checker = TypeChecker()
        errors = checker.check(ast)
        return ast, errors, checker

    return _analyze


@pytest.fixture
def analyze_purity_fixture(parse):
    """Fixture to run purity analysis on source code."""

    def _analyze(source: str) -> tuple[Program, dict[str, PurityInfo]]:
        ast = parse(source)
        purity_info = analyze_purity(ast)
        return ast, purity_info

    return _analyze


@pytest.fixture
def analyze_complexity_fixture(parse):
    """Fixture to run complexity analysis on source code."""

    def _analyze(source: str) -> tuple[Program, dict[str, ComplexityInfo]]:
        ast = parse(source)
        complexity_info = analyze_complexity(ast)
        return ast, complexity_info

    return _analyze


@pytest.fixture
def build_call_graph(parse):
    """Fixture to build call graph from source code."""

    def _build(source: str) -> tuple[Program, CallGraph]:
        ast = parse(source)
        builder = CallGraphBuilder()
        graph = builder.build(ast)
        return ast, graph

    return _build


@pytest.fixture
def analyze_parallelization_fixture(parse):
    """Fixture to analyze parallelization potential in source code."""

    def _analyze(source: str) -> tuple[Program, list[tuple[str, LoopAnalysis]]]:
        ast = parse(source)
        analyzer = ParallelAnalyzer()
        results: list[tuple[str, LoopAnalysis]] = []

        for stmt in ast.statements:
            from mathviz.compiler.ast_nodes import FunctionDef

            if isinstance(stmt, FunctionDef):
                loop_analyses = analyzer.analyze_function(stmt)
                for _loop, analysis in loop_analyses:
                    results.append((stmt.name, analysis))

        return ast, results

    return _analyze
