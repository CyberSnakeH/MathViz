"""
MathViz Compiler Package.

This package contains the core compiler components:
- Lexer: Tokenizes MathViz source code
- Parser: Produces an Abstract Syntax Tree from tokens
- AST: Node definitions for the syntax tree
- CodeGen: Generates Python code from the AST
- TypeChecker: Type inference and checking
- PurityAnalyzer: Analyzes functions for side effects and purity
- ComplexityAnalyzer: Estimates algorithmic complexity (Big-O)
- CallGraph: Analyzes function call relationships and dependencies
- ParallelAnalyzer: Detects parallelizable loops for Numba prange
- CompilationPipeline: Unified compilation interface integrating all analyzers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mathviz.compiler.ast_nodes import ForStatement, FunctionDef, Program, UseStatement
from mathviz.compiler.module_loader import (
    DependencyGraph,
    ModuleInfo,
    ModuleLoader,
    ModuleRegistry,
    is_python_module,
    PYTHON_MODULES,
)
from mathviz.compiler.call_graph import (
    CallGraph,
    CallGraphBuilder,
    CallGraphError,
    CallGraphNode,
    CallSite,
)
from mathviz.compiler.codegen import CodeGenerator
from mathviz.compiler.complexity_analyzer import (
    Complexity,
    ComplexityAnalyzer,
    ComplexityInfo,
    analyze_complexity,
)
from mathviz.compiler.const_fold import (
    AlgebraicSimplifier,
    ConstantFolder,
    ConstantPropagator,
    ConstantScope,
    ConstOptimizer,
    CSEliminator,
    DeadCodeEliminator,
    ExpressionKey,
    OptimizerPass,
    StrengthReducer,
    eliminate_cse,
    eliminate_dead_code,
    fold_constants,
    optimize_program,
    propagate_constants,
    reduce_strength,
    simplify_algebra,
)
from mathviz.compiler.diagnostics import (
    Diagnostic,
    DiagnosticEmitter,
    DiagnosticSeverity,
    SourceSpan,
    find_best_match,
    find_similar_names,
    format_error,
    levenshtein_distance,
)
from mathviz.compiler.jit_optimizer import (
    CacheHints,
    CacheOptimizer,
    CostModel,
    JitAnalyzer,
    JitDecision,
    JitStrategy,
    LoopOptimizer,
    LoopPattern,
    LoopTransformation,
    MemoryAccessPattern,
    MemoryPattern,
    ReductionInfo,
    TilingInfo,
    VectorizableOp,
    VectorizationAnalyzer,
    VectorizationInfo,
    analyze_jit_decision,
    estimate_jit_speedup,
    generate_optimized_function,
    get_loop_optimizations,
    is_numba_compatible,
)
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.linter import (
    ALL_RULES,
    DIVISION_BY_ZERO_POSSIBLE,
    FLOAT_EQUALITY,
    NAMING_CONVENTION,
    NON_EXHAUSTIVE_MATCH,
    REDUNDANT_ASSIGNMENT,
    REDUNDANT_PATTERN,
    RULES_BY_NAME,
    SHADOWING,
    UNREACHABLE_CODE,
    UNREACHABLE_PATTERN,
    UNUSED_FUNCTION,
    UNUSED_IMPORT,
    UNUSED_PARAMETER,
    # Individual rules
    UNUSED_VARIABLE,
    LintCategory,
    LintConfiguration,
    Linter,
    LintLevel,
    LintRule,
    LintViolation,
    get_rule_by_code,
    get_rule_by_name,
    get_rules_by_category,
    lint_program,
    lint_source,
)
from mathviz.compiler.memory_optimizer import (
    AccessPattern,
    AllocationAnalyzer,
    AllocationInfo,
    ArrayAccess,
    BlockingInfo,
    BufferReuseOptimizer,
    CacheAnalysis,
    Graph,
    InPlaceCandidate,
    InPlaceOptimizer,
    LayoutOptimizer,
    LoopInterchange,
    MemoryOptimizer,
    MemoryOrder,
    MemoryPoolGenerator,
    MemoryReport,
    TemporaryEliminator,
    analyze_cache,
    analyze_memory,
    find_allocations,
    find_inplace_candidates,
    generate_memory_pool,
    suggest_buffer_reuse,
)
from mathviz.compiler.memory_optimizer import (
    CacheOptimizer as MemoryCacheOptimizer,
)
from mathviz.compiler.parallel_analyzer import (
    DataDependency,
    DependencyType,
    LoopAnalysis,
    ParallelAnalyzer,
    ReductionOperator,
    ReductionVariable,
    analyze_parallelization,
    can_parallelize_loop,
)
from mathviz.compiler.parser import Parser
from mathviz.compiler.purity_analyzer import (
    Purity,
    PurityAnalyzer,
    PurityInfo,
    SideEffect,
    SideEffectKind,
    analyze_purity,
    can_memoize,
    can_parallelize,
    is_jit_safe,
)
from mathviz.compiler.type_checker import (
    ANY_TYPE,
    ARRAY_TYPE,
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    MAT_TYPE,
    NONE_TYPE,
    STRING_TYPE,
    VEC_TYPE,
    AnyType,
    ArrayType,
    ClassType,
    ConversionInfo,
    FunctionSignature,
    FunctionType,
    GenericTypeInstance,
    PrimitiveType,
    RangeType,
    SymbolTable,
    Type,
    TypeChecker,
    TypeConversion,
    UnknownType,
    infer_expression_type,
    type_check,
)
from mathviz.compiler.vectorizer import (
    ArrayAccessInfo,
    BroadcastAnalyzer,
    BroadcastInfo,
    LoopVectorizer,
    NumpyTransformer,
    SIMDGenerator,
    StencilDetector,
    StencilInfo,
    VectorizationReport,
    VectorizationResult,
    VectorizationStrategy,
    analyze_vectorization,
    can_vectorize_loop,
    generate_vectorized_code,
    vectorize_function,
)
from mathviz.compiler.vectorizer import (
    LoopPattern as VectorizationLoopPattern,
)
from mathviz.compiler.vectorizer import (
    ReductionInfo as VectorizationReductionInfo,
)
from mathviz.compiler.vectorizer import (
    ReductionType as VectorizationReductionType,
)
from mathviz.utils.errors import ModuleResolutionError, SourceLocation
from mathviz.utils.errors import TypeError as MathVizTypeError

# =============================================================================
# Compilation Result Data Structures
# =============================================================================


@dataclass
class FunctionAnalysis:
    """
    Comprehensive analysis results for a single function.

    This dataclass aggregates all analysis information from the various
    analyzers for a single function, providing a unified view of its
    characteristics for optimization and code generation.

    Attributes:
        name: Function name
        purity: Purity analysis information
        complexity: Algorithmic complexity information
        parallelizable_loops: Analysis of loops that can be parallelized
        is_jit_compatible: Whether function can be JIT compiled with Numba
        suggested_optimizations: List of optimization suggestions
        type_signature: Inferred function type signature (if available)
    """

    name: str
    purity: PurityInfo = field(default_factory=PurityInfo)
    complexity: ComplexityInfo | None = None
    parallelizable_loops: list[LoopAnalysis] = field(default_factory=list)
    is_jit_compatible: bool = False
    suggested_optimizations: list[str] = field(default_factory=list)
    type_signature: FunctionSignature | None = None

    def __str__(self) -> str:
        lines = [f"Function: {self.name}"]
        lines.append(f"  Purity: {self.purity.purity.name}")
        if self.complexity:
            lines.append(f"  Complexity: {self.complexity.complexity.value}")
        lines.append(f"  JIT Compatible: {self.is_jit_compatible}")
        if self.parallelizable_loops:
            parallel_count = sum(1 for l in self.parallelizable_loops if l.is_parallelizable)
            lines.append(f"  Parallelizable Loops: {parallel_count}/{len(self.parallelizable_loops)}")
        if self.suggested_optimizations:
            lines.append("  Suggested Optimizations:")
            for opt in self.suggested_optimizations:
                lines.append(f"    - {opt}")
        return "\n".join(lines)


@dataclass
class CompilationResult:
    """
    Complete result of the compilation pipeline.

    This dataclass contains all outputs from the compilation process,
    including generated code, errors, warnings, and analysis results.

    Attributes:
        python_code: Generated Python source code
        type_errors: List of type errors found during type checking
        warnings: List of compilation warnings
        call_graph: The program's call graph
        function_analysis: Per-function analysis results
        ast: The parsed AST (useful for further processing)
        success: Whether compilation completed without fatal errors
    """

    python_code: str
    type_errors: list[MathVizTypeError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    call_graph: CallGraph | None = None
    function_analysis: dict[str, FunctionAnalysis] = field(default_factory=dict)
    ast: Program | None = None
    success: bool = True

    def __str__(self) -> str:
        lines = ["Compilation Result:"]
        lines.append(f"  Success: {self.success}")
        if self.type_errors:
            lines.append(f"  Type Errors: {len(self.type_errors)}")
            for err in self.type_errors[:5]:
                lines.append(f"    - {err.message}")
            if len(self.type_errors) > 5:
                lines.append(f"    ... and {len(self.type_errors) - 5} more")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for warn in self.warnings[:5]:
                lines.append(f"    - {warn}")
        if self.function_analysis:
            lines.append(f"  Functions Analyzed: {len(self.function_analysis)}")
        if self.call_graph:
            lines.append(f"  Call Graph: {len(self.call_graph.nodes)} nodes, {len(self.call_graph.edges)} edges")
        lines.append(f"  Generated Code: {len(self.python_code)} characters")
        return "\n".join(lines)

    def has_errors(self) -> bool:
        """Check if compilation produced any errors."""
        return bool(self.type_errors) or not self.success

    def get_jit_compatible_functions(self) -> list[str]:
        """Get names of all JIT-compatible functions."""
        return [
            name for name, analysis in self.function_analysis.items()
            if analysis.is_jit_compatible
        ]

    def get_parallelizable_functions(self) -> list[str]:
        """Get names of functions with parallelizable loops."""
        return [
            name for name, analysis in self.function_analysis.items()
            if any(loop.is_parallelizable for loop in analysis.parallelizable_loops)
        ]


# =============================================================================
# Compilation Pipeline
# =============================================================================


class CompilationPipeline:
    """
    Unified compilation pipeline that integrates all MathViz analyzers.

    This class provides a single entry point for compiling MathViz source code,
    running all analysis passes, and generating optimized Python code.

    The pipeline performs the following stages:
    1. Lexing - Tokenize source code
    2. Parsing - Build Abstract Syntax Tree
    3. Type Checking - Infer and verify types (optional)
    4. Call Graph Building - Analyze function dependencies
    5. Purity Analysis - Detect side effects
    6. Complexity Analysis - Estimate Big-O complexity
    7. Parallelization Analysis - Detect parallelizable loops
    8. Code Generation - Produce Python code

    Example:
        pipeline = CompilationPipeline(optimize=True, strict=False)
        result = pipeline.compile(source_code)
        if result.success:
            print(result.python_code)
        else:
            for error in result.type_errors:
                print(error)

    Attributes:
        optimize: Enable Numba JIT optimization for compatible functions
        check_types: Enable type checking phase
        analyze_complexity: Enable complexity analysis phase
        auto_parallelize: Enable automatic parallelization detection
        strict: Treat type errors as fatal (stop compilation)
    """

    def __init__(
        self,
        optimize: bool = True,
        check_types: bool = True,
        analyze_complexity: bool = True,
        auto_parallelize: bool = True,
        strict: bool = False,
        module_search_paths: list[Path] | None = None,
    ) -> None:
        """
        Initialize the compilation pipeline.

        Args:
            optimize: Whether to inject Numba JIT decorators for optimization
            check_types: Whether to run the type checking phase
            analyze_complexity: Whether to analyze algorithmic complexity
            auto_parallelize: Whether to detect and optimize parallelizable loops
            strict: If True, type errors prevent code generation
            module_search_paths: Additional directories to search for .mviz modules
        """
        self.optimize = optimize
        self.check_types = check_types
        self.analyze_complexity = analyze_complexity
        self.auto_parallelize = auto_parallelize
        self.strict = strict
        self.module_search_paths = module_search_paths or []

        # Analyzer instances (reusable across compilations)
        self._type_checker: TypeChecker | None = None
        self._purity_analyzer: PurityAnalyzer | None = None
        self._complexity_analyzer: ComplexityAnalyzer | None = None
        self._call_graph_builder: CallGraphBuilder | None = None
        self._parallel_analyzer: ParallelAnalyzer | None = None
        self._module_registry: ModuleRegistry | None = None

    def compile(self, source: str, filename: str = "<string>") -> CompilationResult:
        """
        Execute the full compilation pipeline.

        This method runs all enabled analysis phases and generates Python code.
        The compilation process is:

        1. Lexing - Tokenize the source code
        2. Parsing - Build the AST
        3. Type Checking - Infer types and report errors (if enabled)
        4. Build Call Graph - Analyze function dependencies
        5. Purity Analysis - Determine function side effects
        6. Complexity Analysis - Estimate Big-O (if enabled)
        7. Parallelization Analysis - Find parallelizable loops (if enabled)
        8. Code Generation - Produce Python code (using analysis results)

        Args:
            source: MathViz source code string
            filename: Optional filename for error reporting

        Returns:
            CompilationResult containing generated code, errors, and analysis
        """
        result = CompilationResult(python_code="", success=True)
        warnings: list[str] = []

        # Stage 1: Lexing
        try:
            lexer = Lexer(source)
            tokens = lexer.tokenize()
        except Exception as e:
            result.success = False
            result.warnings.append(f"Lexer error: {e}")
            return result

        # Stage 2: Parsing
        try:
            parser = Parser(tokens)
            ast = parser.parse()
            result.ast = ast
        except Exception as e:
            result.success = False
            result.warnings.append(f"Parser error: {e}")
            return result

        # Stage 2.5: Module Loading
        file_path = Path(filename) if filename != "<string>" else Path.cwd() / "main.mviz"
        root_path = file_path.parent if file_path.exists() else Path.cwd()

        module_loader = ModuleLoader(
            root_path=root_path,
            search_paths=self.module_search_paths,
        )

        # Register the main module
        main_module = module_loader.load_from_ast(ast, "__main__", file_path)

        # Resolve all use statements and load dependencies
        for stmt in ast.statements:
            if isinstance(stmt, UseStatement):
                if not is_python_module(stmt.module_path):
                    try:
                        module_loader.resolve_use_statement(stmt, file_path)
                    except ModuleResolutionError as e:
                        warnings.append(f"Module warning: {e.message}")
                        if self.strict:
                            result.success = False
                            result.warnings.append(str(e))
                            return result

        # Store the module registry for code generation
        self._module_registry = module_loader.registry

        # Stage 3: Type Checking (if enabled)
        type_errors: list[MathVizTypeError] = []
        type_checker: TypeChecker | None = None
        if self.check_types:
            type_checker = TypeChecker()
            type_errors = type_checker.check(ast)
            result.type_errors = type_errors

            if type_errors and self.strict:
                result.success = False
                result.warnings.append(
                    f"Type checking failed with {len(type_errors)} error(s) (strict mode)"
                )
                return result

            if type_errors:
                warnings.append(
                    f"Type checking found {len(type_errors)} error(s)"
                )

        # Stage 4: Build Call Graph
        call_graph_builder = CallGraphBuilder()
        call_graph = call_graph_builder.build(ast)
        result.call_graph = call_graph

        # Check for cycles (warning, not error)
        cycles = call_graph.find_cycles()
        if cycles:
            cycle_names = [" -> ".join(cycle) for cycle in cycles]
            warnings.append(f"Recursive cycles detected: {'; '.join(cycle_names)}")

        # Stage 5: Purity Analysis
        purity_analyzer = PurityAnalyzer()
        purity_results = purity_analyzer.analyze_program(ast)

        # Stage 6: Complexity Analysis (if enabled)
        complexity_results: dict[str, ComplexityInfo] = {}
        if self.analyze_complexity:
            complexity_analyzer = ComplexityAnalyzer()
            complexity_results = complexity_analyzer.analyze_program(ast)

        # Stage 7: Parallelization Analysis (if enabled)
        parallel_results: dict[str, list[tuple[ForStatement, LoopAnalysis]]] = {}
        if self.auto_parallelize:
            parallel_analyzer = ParallelAnalyzer()
            for stmt in ast.statements:
                if isinstance(stmt, FunctionDef):
                    loop_analyses = parallel_analyzer.analyze_function(stmt)
                    if loop_analyses:
                        parallel_results[stmt.name] = loop_analyses

        # Aggregate per-function analysis
        function_analysis = self._aggregate_function_analysis(
            ast=ast,
            purity_results=purity_results,
            complexity_results=complexity_results,
            parallel_results=parallel_results,
            type_checker=type_checker,
        )
        result.function_analysis = function_analysis

        # Generate optimization suggestions
        for func_name, analysis in function_analysis.items():
            self._generate_optimization_suggestions(analysis, warnings)

        # Stage 8: Code Generation
        try:
            generator = CodeGenerator(
                optimize=self.optimize,
                module_registry=self._module_registry,
            )
            python_code = generator.generate(ast)
            result.python_code = python_code
        except Exception as e:
            result.success = False
            result.warnings.append(f"Code generation error: {e}")
            return result

        result.warnings = warnings
        return result

    def compile_file(self, path: Path | str) -> CompilationResult:
        """
        Compile a MathViz file.

        This is a convenience method that reads a file and compiles it.

        Args:
            path: Path to the .mviz file

        Returns:
            CompilationResult containing generated code, errors, and analysis
        """
        path = Path(path)
        if not path.exists():
            result = CompilationResult(python_code="", success=False)
            result.warnings.append(f"File not found: {path}")
            return result

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            result = CompilationResult(python_code="", success=False)
            result.warnings.append(f"Failed to read file: {e}")
            return result

        return self.compile(source, filename=str(path))

    def _aggregate_function_analysis(
        self,
        ast: Program,
        purity_results: dict[str, PurityInfo],
        complexity_results: dict[str, ComplexityInfo],
        parallel_results: dict[str, list[tuple[ForStatement, LoopAnalysis]]],
        type_checker: TypeChecker | None,
    ) -> dict[str, FunctionAnalysis]:
        """
        Aggregate all analysis results into per-function FunctionAnalysis objects.
        """
        function_analysis: dict[str, FunctionAnalysis] = {}

        for stmt in ast.statements:
            if isinstance(stmt, FunctionDef):
                func_name = stmt.name

                # Get purity info
                purity_info = purity_results.get(func_name, PurityInfo())

                # Get complexity info
                complexity_info = complexity_results.get(func_name)

                # Get parallelization analysis
                loop_analyses: list[LoopAnalysis] = []
                if func_name in parallel_results:
                    loop_analyses = [
                        analysis for _, analysis in parallel_results[func_name]
                    ]

                # Determine JIT compatibility
                is_jit_compatible = is_jit_safe(purity_info) and not purity_info.has_manim_calls()

                # Get type signature if available
                type_signature: FunctionSignature | None = None
                if type_checker and func_name in type_checker.function_signatures:
                    type_signature = type_checker.function_signatures[func_name]

                function_analysis[func_name] = FunctionAnalysis(
                    name=func_name,
                    purity=purity_info,
                    complexity=complexity_info,
                    parallelizable_loops=loop_analyses,
                    is_jit_compatible=is_jit_compatible,
                    suggested_optimizations=[],
                    type_signature=type_signature,
                )

        return function_analysis

    def _generate_optimization_suggestions(
        self,
        analysis: FunctionAnalysis,
        warnings: list[str],
    ) -> None:
        """
        Generate optimization suggestions based on analysis results.
        """
        suggestions = analysis.suggested_optimizations

        # JIT optimization suggestions
        if analysis.is_jit_compatible:
            if analysis.purity.is_pure():
                if can_memoize(analysis.purity):
                    suggestions.append(
                        f"Function '{analysis.name}' is pure and can be memoized for repeated calls"
                    )
        else:
            if analysis.purity.has_io():
                suggestions.append(
                    f"Function '{analysis.name}' has I/O operations; "
                    "extract pure computation for JIT optimization"
                )
            if analysis.purity.has_manim_calls():
                suggestions.append(
                    f"Function '{analysis.name}' contains Manim calls; "
                    "move to scene methods"
                )

        # Complexity-based suggestions
        if analysis.complexity:
            if analysis.complexity.complexity in (Complexity.O_N_SQUARED, Complexity.O_N_CUBED):
                suggestions.append(
                    f"Function '{analysis.name}' has {analysis.complexity.complexity.value} "
                    "complexity; consider algorithmic improvements"
                )
            if analysis.complexity.complexity == Complexity.O_2_N:
                suggestions.append(
                    f"Function '{analysis.name}' has exponential complexity; "
                    "consider dynamic programming or memoization"
                )

        # Parallelization suggestions
        for loop_analysis in analysis.parallelizable_loops:
            if loop_analysis.is_parallelizable:
                if loop_analysis.reduction_vars:
                    suggestions.append(
                        f"Loop in '{analysis.name}' can be parallelized with "
                        f"reduction variables: {', '.join(loop_analysis.reduction_vars)}"
                    )
                else:
                    suggestions.append(
                        f"Loop in '{analysis.name}' can be parallelized with prange"
                    )
            else:
                for transform in loop_analysis.suggested_transforms:
                    suggestions.append(
                        f"In '{analysis.name}': {transform}"
                    )


# =============================================================================
# Convenience Functions (Updated)
# =============================================================================


def compile_source(source: str, optimize: bool = True) -> str:
    """
    Compile MathViz source code to Python.

    This is a simplified convenience function that uses the CompilationPipeline
    with default settings. For more control, use CompilationPipeline directly.

    Args:
        source: MathViz source code string
        optimize: Whether to inject Numba JIT decorators for optimization

    Returns:
        Generated Python source code

    Raises:
        ValueError: If compilation fails with strict type errors
    """
    pipeline = CompilationPipeline(
        optimize=optimize,
        check_types=True,
        analyze_complexity=False,  # Skip for simple compilation
        auto_parallelize=False,    # Skip for simple compilation
        strict=False,
    )
    result = pipeline.compile(source)

    if not result.success:
        error_msg = "; ".join(result.warnings)
        raise ValueError(f"Compilation failed: {error_msg}")

    return result.python_code


def compile_file(filepath: str, optimize: bool = True) -> str:
    """
    Compile a MathViz file to Python.

    This is a simplified convenience function that uses the CompilationPipeline
    with default settings. For more control, use CompilationPipeline directly.

    Args:
        filepath: Path to the .mviz file
        optimize: Whether to inject Numba JIT decorators for optimization

    Returns:
        Generated Python source code

    Raises:
        ValueError: If compilation fails
    """
    pipeline = CompilationPipeline(
        optimize=optimize,
        check_types=True,
        analyze_complexity=False,
        auto_parallelize=False,
        strict=False,
    )
    result = pipeline.compile_file(filepath)

    if not result.success:
        error_msg = "; ".join(result.warnings)
        raise ValueError(f"Compilation failed: {error_msg}")

    return result.python_code


def compile_with_analysis(
    source: str,
    filename: str = "<string>",
    optimize: bool = True,
    strict: bool = False,
) -> CompilationResult:
    """
    Compile MathViz source code with full analysis.

    This function runs the complete compilation pipeline including all
    analysis phases and returns detailed results.

    Args:
        source: MathViz source code string
        filename: Optional filename for error reporting
        optimize: Whether to enable JIT optimization
        strict: Whether to treat type errors as fatal

    Returns:
        CompilationResult with generated code and analysis
    """
    pipeline = CompilationPipeline(
        optimize=optimize,
        check_types=True,
        analyze_complexity=True,
        auto_parallelize=True,
        strict=strict,
    )
    return pipeline.compile(source, filename)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core compiler
    "Lexer",
    "Parser",
    "CodeGenerator",
    "Program",
    "compile_source",
    "compile_file",
    "compile_with_analysis",
    # Compilation Pipeline
    "CompilationPipeline",
    "CompilationResult",
    "FunctionAnalysis",
    # Module system
    "ModuleLoader",
    "ModuleInfo",
    "ModuleRegistry",
    "DependencyGraph",
    "is_python_module",
    "PYTHON_MODULES",
    "ModuleResolutionError",
    # Type system
    "TypeChecker",
    "Type",
    "PrimitiveType",
    "GenericTypeInstance",
    "ArrayType",
    "FunctionType",
    "ClassType",
    "UnknownType",
    "AnyType",
    "RangeType",
    "FunctionSignature",
    "SymbolTable",
    "TypeConversion",
    "ConversionInfo",
    "type_check",
    "infer_expression_type",
    "INT_TYPE",
    "FLOAT_TYPE",
    "BOOL_TYPE",
    "STRING_TYPE",
    "NONE_TYPE",
    "VEC_TYPE",
    "MAT_TYPE",
    "ARRAY_TYPE",
    "ANY_TYPE",
    # Purity analysis
    "PurityAnalyzer",
    "Purity",
    "PurityInfo",
    "SideEffect",
    "SideEffectKind",
    "analyze_purity",
    "is_jit_safe",
    "can_memoize",
    "can_parallelize",
    # Complexity analysis
    "ComplexityAnalyzer",
    "Complexity",
    "ComplexityInfo",
    "analyze_complexity",
    # Call graph
    "CallGraph",
    "CallGraphBuilder",
    "CallGraphNode",
    "CallSite",
    "CallGraphError",
    # Parallel analysis
    "ParallelAnalyzer",
    "LoopAnalysis",
    "DataDependency",
    "DependencyType",
    "ReductionOperator",
    "ReductionVariable",
    "analyze_parallelization",
    "can_parallelize_loop",
    # JIT optimizer
    "JitAnalyzer",
    "JitStrategy",
    "JitDecision",
    "LoopOptimizer",
    "VectorizationAnalyzer",
    "CacheOptimizer",
    "CostModel",
    "MemoryAccessPattern",
    "LoopPattern",
    "VectorizableOp",
    "LoopTransformation",
    "ReductionInfo",
    "TilingInfo",
    "MemoryPattern",
    "VectorizationInfo",
    "CacheHints",
    "analyze_jit_decision",
    "get_loop_optimizations",
    "estimate_jit_speedup",
    "is_numba_compatible",
    "generate_optimized_function",
    # Vectorizer (SIMD Vectorization Pass)
    "VectorizationStrategy",
    "VectorizationLoopPattern",
    "VectorizationReductionType",
    "VectorizationResult",
    "ArrayAccessInfo",
    "VectorizationReductionInfo",
    "StencilInfo",
    "BroadcastInfo",
    "LoopVectorizer",
    "NumpyTransformer",
    "StencilDetector",
    "BroadcastAnalyzer",
    "SIMDGenerator",
    "VectorizationReport",
    "vectorize_function",
    "generate_vectorized_code",
    "analyze_vectorization",
    "can_vectorize_loop",
    # Linter
    "Linter",
    "LintRule",
    "LintViolation",
    "LintLevel",
    "LintCategory",
    "LintConfiguration",
    "lint_source",
    "lint_program",
    "get_rule_by_name",
    "get_rule_by_code",
    "get_rules_by_category",
    "ALL_RULES",
    "RULES_BY_NAME",
    "UNUSED_VARIABLE",
    "UNUSED_FUNCTION",
    "UNUSED_PARAMETER",
    "UNUSED_IMPORT",
    "UNREACHABLE_CODE",
    "UNREACHABLE_PATTERN",
    "NON_EXHAUSTIVE_MATCH",
    "REDUNDANT_PATTERN",
    "REDUNDANT_ASSIGNMENT",
    "FLOAT_EQUALITY",
    "DIVISION_BY_ZERO_POSSIBLE",
    "SHADOWING",
    "NAMING_CONVENTION",
    # Diagnostics
    "DiagnosticSeverity",
    "SourceSpan",
    "Diagnostic",
    "DiagnosticEmitter",
    "levenshtein_distance",
    "find_similar_names",
    "find_best_match",
    "format_error",
    # Constant folding and optimization
    "OptimizerPass",
    "ConstantFolder",
    "ConstantPropagator",
    "DeadCodeEliminator",
    "AlgebraicSimplifier",
    "StrengthReducer",
    "CSEliminator",
    "ConstOptimizer",
    "ConstantScope",
    "ExpressionKey",
    "fold_constants",
    "propagate_constants",
    "eliminate_dead_code",
    "simplify_algebra",
    "reduce_strength",
    "eliminate_cse",
    "optimize_program",
    # Memory optimizer
    "MemoryOptimizer",
    "AllocationAnalyzer",
    "AllocationInfo",
    "BufferReuseOptimizer",
    "InPlaceOptimizer",
    "InPlaceCandidate",
    "MemoryCacheOptimizer",
    "CacheAnalysis",
    "LayoutOptimizer",
    "TemporaryEliminator",
    "MemoryPoolGenerator",
    "MemoryReport",
    "ArrayAccess",
    "AccessPattern",
    "MemoryOrder",
    "LoopInterchange",
    "BlockingInfo",
    "Graph",
    "analyze_memory",
    "find_allocations",
    "suggest_buffer_reuse",
    "find_inplace_candidates",
    "analyze_cache",
    "generate_memory_pool",
]
