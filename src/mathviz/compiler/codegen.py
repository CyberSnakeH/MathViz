"""
MathViz Code Generator.

Transforms an AST into executable Python code, handling:
- Mathematical operator translation (∈ -> in, ∪ -> union, etc.)
- Manim scene generation
- Optional Numba JIT decorator injection for optimizable functions
- Proper Python syntax with correct indentation
- Purity-based JIT decisions using PurityAnalyzer results
- Automatic prange insertion using ParallelAnalyzer results
- Complexity annotations for verbose output
- Function ordering based on call graph analysis
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mathviz.compiler.purity_analyzer import PurityInfo
    from mathviz.compiler.complexity_analyzer import ComplexityInfo
    from mathviz.compiler.parallel_analyzer import LoopAnalysis
    from mathviz.compiler.call_graph import CallGraph
    from mathviz.compiler.module_loader import ModuleInfo, ModuleRegistry

from mathviz.compiler.module_loader import PYTHON_MODULES

from mathviz.compiler.ast_nodes import (
    # Program
    Program,
    Block,
    # Types
    TypeAnnotation,
    SimpleType,
    GenericType,
    FunctionType,
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
    SomeExpression,
    OkExpression,
    ErrExpression,
    UnwrapExpression,
    AwaitExpression,
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    KeywordArgument,
    # Pattern matching
    Pattern,
    LiteralPattern,
    IdentifierPattern,
    TuplePattern,
    ConstructorPattern,
    RangePattern,
    OrPattern,
    BindingPattern,
    RestPattern,
    ListPattern,
    MatchArm,
    MatchExpression,
    MemberAccess,
    IndexExpression,
    ConditionalExpression,
    LambdaExpression,
    RangeExpression,
    BinaryOperator,
    UnaryOperator,
    # JIT/Numba
    JitMode,
    JitOptions,
    # F-strings
    FStringPart,
    FStringLiteral,
    FStringExpression,
    FString,
    # Statements
    Statement,
    ExpressionStatement,
    LetStatement,
    DestructuringLetStatement,
    ConstDeclaration,
    AssignmentStatement,
    CompoundAssignment,
    FunctionDef,
    AsyncFunctionDef,
    ClassDef,
    SceneDef,
    IfStatement,
    ForStatement,
    AsyncForStatement,
    WhileStatement,
    LoopStatement,
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
    BaseASTVisitor,
    # OOP constructs
    Visibility,
    StructField,
    StructDef,
    Method,
    ImplBlock,
    TraitMethod,
    TraitDef,
    EnumVariant,
    EnumDef,
    SelfExpression,
    EnumVariantAccess,
    StructLiteral,
    EnumPattern,
    # Comprehensions and pipe lambdas
    ComprehensionClause,
    ListComprehension,
    SetComprehension,
    DictComprehension,
    PipeLambda,
    # Generic type parameters
    TypeParameter,
    WhereClause,
)


# Binary operator to Python code mapping
BINARY_OP_TO_PYTHON: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "+",
    BinaryOperator.SUB: "-",
    BinaryOperator.MUL: "*",
    BinaryOperator.DIV: "/",
    BinaryOperator.FLOOR_DIV: "//",
    BinaryOperator.MOD: "%",
    BinaryOperator.POW: "**",
    BinaryOperator.EQ: "==",
    BinaryOperator.NE: "!=",
    BinaryOperator.LT: "<",
    BinaryOperator.GT: ">",
    BinaryOperator.LE: "<=",
    BinaryOperator.GE: ">=",
    BinaryOperator.AND: "and",
    BinaryOperator.OR: "or",
    BinaryOperator.IN: "in",
}

# Unary operator to Python code mapping
UNARY_OP_TO_PYTHON: dict[UnaryOperator, str] = {
    UnaryOperator.NEG: "-",
    UnaryOperator.POS: "+",
    UnaryOperator.NOT: "not ",
}

# MathViz type to Python type mapping
TYPE_TO_PYTHON: dict[str, str] = {
    "Int": "int",
    "Float": "float",
    "Bool": "bool",
    "String": "str",
    "List": "list",
    "Set": "set",
    "Dict": "dict",
    "Tuple": "tuple",
    "Optional": "Optional",
    "Result": "Result",
    # NumPy types for JIT compatibility
    "Array": "np.ndarray",
    "Vec": "np.ndarray",
    "Mat": "np.ndarray",
    "Matrix": "np.ndarray",
    # Rust-style type aliases
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "f32": "float",
    "f64": "float",
    "bool": "bool",
    "str": "str",
}

# Functions that should be converted to numpy calls
NUMPY_FUNCTION_MAP = {
    # Array creation
    "zeros": "np.zeros",
    "ones": "np.ones",
    "empty": "np.empty",
    "arange": "np.arange",
    "linspace": "np.linspace",
    "eye": "np.eye",
    "identity": "np.identity",
    "full": "np.full",
    # Math operations
    "sqrt": "np.sqrt",
    "sin": "np.sin",
    "cos": "np.cos",
    "tan": "np.tan",
    "exp": "np.exp",
    "log": "np.log",
    "log10": "np.log10",
    "log2": "np.log2",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "abs": "np.abs",
    "fabs": "np.fabs",
    "asin": "np.arcsin",
    "acos": "np.arccos",
    "atan": "np.arctan",
    "atan2": "np.arctan2",
    "sinh": "np.sinh",
    "cosh": "np.cosh",
    "tanh": "np.tanh",
    # Array operations
    "dot": "np.dot",
    "matmul": "np.matmul",
    "transpose": "np.transpose",
    "reshape": "np.reshape",
    "flatten": "np.ravel",
    "concatenate": "np.concatenate",
    "stack": "np.stack",
    "vstack": "np.vstack",
    "hstack": "np.hstack",
    "sum": "np.sum",
    "mean": "np.mean",
    "std": "np.std",
    "var": "np.var",
    "min": "np.min",
    "max": "np.max",
    "argmax": "np.argmax",
    "argmin": "np.argmin",
    "argsort": "np.argsort",
    "sort": "np.sort",
    "clip": "np.clip",
    "where": "np.where",
    "nonzero": "np.nonzero",
    "unique": "np.unique",
    "diff": "np.diff",
    "cumsum": "np.cumsum",
    "cumprod": "np.cumprod",
}

# Built-in I/O functions
BUILTIN_IO_FUNCTIONS = {
    "read_file": "_mviz_read_file",
    "write_file": "_mviz_write_file",
    "append_file": "_mviz_append_file",
    "file_exists": "_mviz_file_exists",
}

# Iterator method mappings - method name to (runtime_func, takes_lambda, arg_count)
# takes_lambda: does the first arg need to be a lambda?
# arg_count: number of additional args (not counting the iterable or lambda)
ITERATOR_METHOD_MAP: dict[str, tuple[str, bool, int]] = {
    # Transformation methods (take lambda)
    "map": ("iter_map", True, 0),
    "filter": ("iter_filter", True, 0),
    "reduce": ("iter_reduce", False, 2),  # (initial, func)
    "fold": ("iter_fold", False, 2),  # (initial, func)
    "flat_map": ("iter_flat_map", True, 0),
    "flatten": ("iter_flatten", False, 0),
    # Access methods
    "first": ("iter_first", False, 0),
    "last": ("iter_last", False, 0),
    "nth": ("iter_nth", False, 1),  # (n)
    "find": ("iter_find", True, 0),
    "position": ("iter_position", True, 0),
    # Predicate methods
    "any": ("iter_any", True, 0),
    "all": ("iter_all", True, 0),
    "none": ("iter_none", True, 0),
    "count": ("iter_count", False, 0),
    "count_if": ("iter_count_if", True, 0),
    # Numeric methods (no lambda)
    "sum": ("iter_sum", False, 0),
    "product": ("iter_product", False, 0),
    "min": ("iter_min", False, 0),
    "max": ("iter_max", False, 0),
    "average": ("iter_average", False, 0),
    "min_by": ("iter_min_by", True, 0),
    "max_by": ("iter_max_by", True, 0),
    # Slicing methods
    "take": ("iter_take", False, 1),  # (n)
    "skip": ("iter_skip", False, 1),  # (n)
    "take_while": ("iter_take_while", True, 0),
    "skip_while": ("iter_skip_while", True, 0),
    # Ordering methods
    "sorted": ("iter_sorted", False, 0),
    "sorted_by": ("iter_sorted_by", True, 0),
    "sorted_by_desc": ("iter_sorted_by_desc", True, 0),
    "reversed": ("iter_reversed", False, 0),
    # Combination methods
    "zip": ("iter_zip", False, 1),  # (other)
    "enumerate": ("iter_enumerate", False, 0),
    "chain": ("iter_chain", False, -1),  # variadic
    "chunk": ("iter_chunk", False, 1),  # (size)
    "unique": ("iter_unique", False, 0),
    # Collection methods
    "collect": ("iter_collect_list", False, 0),
    "collect_list": ("iter_collect_list", False, 0),
    "collect_set": ("iter_collect_set", False, 0),
    "collect_dict": ("iter_collect_dict", False, 0),
    "join": ("iter_join", False, 1),  # (separator)
    "group_by": ("iter_group_by", True, 0),
    "partition": ("iter_partition", True, 0),
    # Dict-specific methods
    "keys": ("dict_keys", False, 0),
    "values": ("dict_values", False, 0),
    "items": ("dict_items", False, 0),
    "map_values": ("dict_map_values", True, 0),
    "filter_keys": ("dict_filter_keys", True, 0),
    "filter_values": ("dict_filter_values", True, 0),
}

# MathViz runtime I/O code to inject
MATHVIZ_IO_RUNTIME = '''
def _mviz_read_file(path: str) -> str:
    """Read entire file contents as string."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _mviz_write_file(path: str, content: str) -> None:
    """Write string content to file (overwrites)."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def _mviz_append_file(path: str, content: str) -> None:
    """Append string content to file."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)

def _mviz_file_exists(path: str) -> bool:
    """Check if file exists."""
    import os
    return os.path.exists(path)

def _mviz_unwrap(value):
    """
    Unwrap an Optional or Result value.

    For Optional (regular values): raises if None
    For Result (tuple): raises if Err, returns Ok value
    """
    # Check if it's a Result type (tuple with boolean flag)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], bool):
        is_ok, inner = value
        if is_ok:
            return inner
        raise RuntimeError(f"Unwrap called on Err: {inner}")
    # Otherwise it's an Optional - just check for None
    if value is None:
        raise RuntimeError("Unwrap called on None")
    return value

def _mviz_unwrap_or(value, default):
    """
    Unwrap an Optional or Result value, returning default on failure.
    """
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], bool):
        is_ok, inner = value
        return inner if is_ok else default
    return value if value is not None else default

def _mviz_is_some(value) -> bool:
    """Check if an Optional value is Some (not None)."""
    return value is not None

def _mviz_is_ok(value) -> bool:
    """Check if a Result is Ok."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], bool):
        return value[0]
    return True  # Non-Result values are considered Ok

def _mviz_is_err(value) -> bool:
    """Check if a Result is Err."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], bool):
        return not value[0]
    return False
'''

# Compound assignment operator mapping
COMPOUND_OP_TO_PYTHON: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "+=",
    BinaryOperator.SUB: "-=",
    BinaryOperator.MUL: "*=",
    BinaryOperator.DIV: "/=",
}


# Types that are compatible with Numba JIT
JIT_COMPATIBLE_TYPES = {"Int", "Float", "Bool", "int", "float", "bool"}
JIT_COMPATIBLE_ARRAY_TYPES = {"List", "Array", "Vec", "Mat", "Matrix", "NDArray"}

# Direct numpy array types (always JIT-compatible)
NUMPY_ARRAY_TYPES = {"Array", "Vec", "Mat", "Matrix", "NDArray"}

# Built-in functions that are JIT-compatible
JIT_COMPATIBLE_BUILTINS = {
    "abs", "min", "max", "sum", "len", "range", "enumerate",
    "int", "float", "bool", "round", "pow",
    # Math functions (numpy)
    "sqrt", "sin", "cos", "tan", "exp", "log", "log10", "log2",
    "floor", "ceil", "fabs", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "degrees", "radians",
    # NumPy array functions
    "zeros", "ones", "empty", "arange", "linspace", "eye",
    "dot", "matmul", "transpose", "reshape", "flatten",
    "concatenate", "stack", "vstack", "hstack",
    "mean", "std", "var", "median", "percentile",
    "argmax", "argmin", "argsort", "sort", "clip",
    "where", "nonzero", "unique", "diff", "cumsum", "cumprod",
}

# NumPy type mappings for Numba compatibility
NUMPY_TYPE_MAP = {
    "Int": "np.int64",
    "Float": "np.float64",
    "Bool": "np.bool_",
    "Complex": "np.complex128",
}

# NumPy dtype strings for array creation
NUMPY_DTYPE_MAP = {
    "Int": "np.int64",
    "Float": "np.float64",
    "Bool": "np.bool_",
    "Complex": "np.complex128",
}


class JitCompatibilityAnalyzer(BaseASTVisitor):
    """
    Analyzes a function to determine if it's compatible with Numba JIT.

    A function is JIT-compatible if:
    - All parameters have numeric types (Int, Float, Bool) or arrays of numeric types
    - Return type is numeric or array of numeric
    - Body only uses numeric operations and JIT-compatible built-ins
    - No string operations, set operations, or object method calls
    - No unsupported Python features (exceptions, generators, etc.)
    """

    def __init__(self) -> None:
        self.is_compatible = True
        self.reasons: list[str] = []
        self._in_function = False

    def analyze(self, func: FunctionDef) -> tuple[bool, list[str]]:
        """Analyze a function for JIT compatibility."""
        self.is_compatible = True
        self.reasons = []
        self._in_function = True

        # Check if already has JIT decorator (explicit override)
        if func.jit_options.mode != JitMode.NONE:
            return True, []

        # Check parameter types
        for param in func.parameters:
            if param.type_annotation:
                if not self._is_jit_compatible_type(param.type_annotation):
                    self.is_compatible = False
                    self.reasons.append(f"Parameter '{param.name}' has non-JIT type")
            # If no type annotation, we can't be sure - still try to JIT

        # Check return type
        if func.return_type:
            if not self._is_jit_compatible_type(func.return_type):
                self.is_compatible = False
                self.reasons.append("Return type is not JIT-compatible")

        # Analyze function body
        if func.body:
            self.visit(func.body)

        self._in_function = False
        return self.is_compatible, self.reasons

    def _is_jit_compatible_type(self, type_ann: TypeAnnotation) -> bool:
        """Check if a type annotation is JIT-compatible."""
        if isinstance(type_ann, SimpleType):
            # Basic numeric types
            if type_ann.name in JIT_COMPATIBLE_TYPES:
                return True
            # NumPy array types (Vec, Mat, Array, etc.)
            if type_ann.name in NUMPY_ARRAY_TYPES:
                return True
            return False
        elif isinstance(type_ann, GenericType):
            # List[Int], List[Float], Array[Float], etc.
            if type_ann.base in JIT_COMPATIBLE_ARRAY_TYPES:
                return all(self._is_jit_compatible_type(arg) for arg in type_ann.type_args)
            return False
        return False

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        """Check binary operations for JIT compatibility."""
        # Set operations are not JIT-compatible
        if node.operator in {
            BinaryOperator.ELEMENT_OF,
            BinaryOperator.NOT_ELEMENT_OF,
            BinaryOperator.SUBSET,
            BinaryOperator.SUPERSET,
            BinaryOperator.PROPER_SUBSET,
            BinaryOperator.PROPER_SUPERSET,
            BinaryOperator.UNION,
            BinaryOperator.INTERSECTION,
            BinaryOperator.SET_DIFF,
        }:
            self.is_compatible = False
            self.reasons.append("Set operations are not JIT-compatible")
        super().visit_binary_expression(node)

    def visit_string_literal(self, node: StringLiteral) -> None:
        """String literals make function non-JIT-compatible."""
        self.is_compatible = False
        self.reasons.append("String literals are not JIT-compatible")

    def visit_set_literal(self, node: SetLiteral) -> None:
        """Set literals are not JIT-compatible."""
        self.is_compatible = False
        self.reasons.append("Set literals are not JIT-compatible")

    def visit_dict_literal(self, node: DictLiteral) -> None:
        """Dict literals are not JIT-compatible."""
        self.is_compatible = False
        self.reasons.append("Dict literals are not JIT-compatible")

    def visit_call_expression(self, node: CallExpression) -> None:
        """Check if function calls are JIT-compatible."""
        # Check if it's a known JIT-compatible builtin
        if isinstance(node.callee, Identifier):
            if node.callee.name not in JIT_COMPATIBLE_BUILTINS:
                # Could be a user-defined function - we'll allow it
                # (Numba can inline JIT functions)
                pass
        elif isinstance(node.callee, MemberAccess):
            # Method calls like obj.method() - might not be JIT-compatible
            # But allow common numpy/math methods
            pass
        super().visit_call_expression(node)


class CodeGenerator(BaseASTVisitor):
    """
    Generates Python code from a MathViz AST.

    The generator produces clean, readable Python code with:
    - Proper indentation
    - Type hints (when available)
    - Runtime library calls for mathematical operators
    - Manim Scene base class for scene definitions
    - AUTOMATIC Numba JIT optimization for compatible functions
    - Purity-based JIT optimization using analyzer results
    - Automatic prange insertion for parallelizable loops
    - Complexity comments in verbose mode
    - Call graph-based function ordering

    Usage:
        generator = CodeGenerator(optimize=True)
        python_code = generator.generate(ast)

        # With analysis data for smarter optimizations:
        generator = CodeGenerator(
            optimize=True,
            purity_info=purity_results,
            parallel_info=parallel_results,
            complexity_info=complexity_results,
            call_graph=call_graph,
        )
        python_code = generator.generate(ast)
    """

    def __init__(
        self,
        optimize: bool = True,
        indent_size: int = 4,
        purity_info: dict[str, PurityInfo] | None = None,
        complexity_info: dict[str, ComplexityInfo] | None = None,
        parallel_info: dict[str, list[LoopAnalysis]] | None = None,
        call_graph: CallGraph | None = None,
        verbose: bool = False,
        module_registry: "ModuleRegistry | None" = None,
    ) -> None:
        """
        Initialize the code generator.

        Args:
            optimize: Whether to automatically inject Numba JIT decorators
            indent_size: Number of spaces per indentation level
            purity_info: Purity analysis results from PurityAnalyzer, keyed by function name.
                        If provided, enables smarter JIT decisions based on function purity.
            complexity_info: Complexity analysis results from ComplexityAnalyzer, keyed by
                            function name. Used to generate complexity comments when verbose=True.
            parallel_info: Parallelization analysis results from ParallelAnalyzer, keyed by
                          function name. Each function maps to a list of LoopAnalysis for
                          its for-loops. Used to auto-insert prange for parallelizable loops.
            call_graph: Call graph from CallGraphBuilder. Used to determine optimal function
                       definition order (callees before callers) and handle mutual recursion.
            verbose: Whether to emit complexity comments before function definitions.
            module_registry: Registry of loaded MathViz modules for code generation.
        """
        self.optimize = optimize
        self.indent_size = indent_size
        self.verbose = verbose

        # Analysis data from analyzer modules (optional, for smarter optimization)
        self._purity_info: dict[str, PurityInfo] = purity_info or {}
        self._complexity_info: dict[str, ComplexityInfo] = complexity_info or {}
        self._parallel_info: dict[str, list[LoopAnalysis]] = parallel_info or {}
        self._call_graph: CallGraph | None = call_graph
        self._module_registry = module_registry

        # Internal state
        self._indent_level = 0
        self._output: list[str] = []
        self._needs_runtime_import = False
        self._needs_numba_import = False
        self._needs_numpy_import = False
        self._needs_manim_import = False
        self._needs_io_runtime = False
        self._needs_iterator_runtime = False  # Track if we need iterator runtime import
        self._needs_prange = False  # Track if we need prange import
        self._needs_asyncio_import = False  # Track if we need asyncio import
        self._in_async_function = False  # Track if we're in an async function
        self._in_scene = False
        self._in_jit_function = False  # Track if we're in a JIT function
        self._current_function: str | None = None  # Track current function for parallel info lookup
        self._current_loop_index = 0  # Track which loop we're generating in current function
        self._jit_analyzer = JitCompatibilityAnalyzer()  # Fallback if no purity info
        self._has_main = False
        self._has_scene = False
        # OOP import flags
        self._needs_dataclass_import = False
        self._needs_dataclass_replace = False  # For struct spread syntax
        self._needs_abc_import = False
        self._needs_enum_import = False
        # Generic type support
        self._needs_typing_import = False
        self._type_vars_declared: set[str] = set()  # Track declared TypeVars
        # Track generated modules to avoid duplicates
        self._generated_modules: set[str] = set()
        # Track static methods from impl blocks: (type_name, method_name)
        self._static_methods: set[tuple[str, str]] = set()
        # Track instance methods from impl blocks: (type_name, method_name)
        self._instance_methods: set[tuple[str, str]] = set()
        # Track struct types for instance method detection
        self._struct_types: set[str] = set()
        # Track enum types and their variants: enum_name -> set of variant names
        self._enum_variants: dict[str, set[str]] = {}
        # Track enum variants that have data (fields)
        self._enum_variants_with_data: set[tuple[str, str]] = set()
        # Track which enums use Python Enum (simple enums without any data)
        self._simple_enums: set[str] = set()

    def generate(self, program: Program) -> str:
        """
        Generate Python code from the AST.

        Args:
            program: The root Program node

        Returns:
            Generated Python source code
        """
        self._indent_level = 0
        self._output = []
        self._needs_runtime_import = False
        self._needs_numba_import = False
        self._needs_prange = False

        # First pass: collect import needs
        self._analyze_imports(program)

        # Generate header comment
        self._emit("# Generated by MathViz Compiler")
        self._emit("# https://github.com/CyberSnakeH/MathViz")
        self._emit("")

        # Generate necessary imports
        if self._needs_runtime_import:
            self._emit("from mathviz.runtime import (")
            self._indent()
            self._emit("set_union,")
            self._emit("set_intersection,")
            self._emit("set_difference,")
            self._emit("is_subset,")
            self._emit("is_superset,")
            self._emit("is_element,")
            self._emit("is_not_element,")
            self._dedent()
            self._emit(")")
            self._emit("")

        if self._needs_numpy_import:
            self._emit("import numpy as np")
            self._emit("")

        if self._needs_numba_import:
            # Include prange if we have parallel loops
            if self._needs_prange:
                self._emit("from numba import jit, njit, vectorize, prange")
            else:
                self._emit("from numba import jit, njit, vectorize, prange")
            self._emit("import numba")
            self._emit("")

        if self._needs_manim_import:
            self._emit("from manim import *")
            self._emit("")

        if self._needs_io_runtime:
            for line in MATHVIZ_IO_RUNTIME.strip().split('\n'):
                self._output.append(line)
            self._emit("")

        # OOP imports
        if self._needs_dataclass_import or self._needs_dataclass_replace:
            imports = []
            if self._needs_dataclass_import:
                imports.append("dataclass")
            if self._needs_dataclass_replace:
                imports.append("replace")
            self._emit(f"from dataclasses import {', '.join(imports)}")
            self._emit("")

        if self._needs_abc_import:
            self._emit("from abc import ABC, abstractmethod")
            self._emit("")

        if self._needs_enum_import:
            self._emit("from enum import Enum, auto")
            self._emit("")

        # Generic type support imports
        if self._needs_typing_import:
            self._emit("from typing import TypeVar, Generic")
            self._emit("")

        # Async support imports
        if self._needs_asyncio_import:
            self._emit("import asyncio")
            self._emit("")

        # Iterator runtime imports
        if self._needs_iterator_runtime:
            self._emit("from mathviz.runtime.iterators import (")
            self._indent()
            # Group imports by category
            self._emit("# Transformation")
            self._emit("iter_map, iter_filter, iter_reduce, iter_fold,")
            self._emit("iter_flat_map, iter_flatten,")
            self._emit("# Access")
            self._emit("iter_first, iter_last, iter_nth, iter_find, iter_position,")
            self._emit("# Predicate")
            self._emit("iter_any, iter_all, iter_none, iter_count, iter_count_if,")
            self._emit("# Numeric")
            self._emit("iter_sum, iter_product, iter_min, iter_max, iter_average,")
            self._emit("iter_min_by, iter_max_by,")
            self._emit("# Slicing")
            self._emit("iter_take, iter_skip, iter_take_while, iter_skip_while,")
            self._emit("# Ordering")
            self._emit("iter_sorted, iter_sorted_by, iter_sorted_by_desc, iter_reversed,")
            self._emit("# Combination")
            self._emit("iter_zip, iter_enumerate, iter_chain, iter_chunk, iter_unique,")
            self._emit("# Collection")
            self._emit("iter_collect_list, iter_collect_set, iter_collect_dict,")
            self._emit("iter_join, iter_group_by, iter_partition,")
            self._emit("# Dict-specific")
            self._emit("dict_keys, dict_values, dict_items,")
            self._emit("dict_map_values, dict_filter_keys, dict_filter_values,")
            self._dedent()
            self._emit(")")
            self._emit("")

        # Generate program body with call graph ordering if available
        self._generate_program_body(program)

        # Add main entry point if main function exists
        if self._has_main:
            self._emit("")
            self._emit("")
            self._emit('if __name__ == "__main__":')
            self._indent()
            self._emit("main()")
            self._dedent()

        return "\n".join(self._output)

    def _generate_program_body(self, program: Program) -> None:
        """
        Generate the program body, ordering functions based on call graph if available.

        When a call graph is provided, functions are ordered so that callees are
        defined before callers (where possible). For mutual recursion, the order
        is preserved from the original source.

        Args:
            program: The root Program node
        """
        if self._call_graph is None:
            # No call graph - use original order
            self.visit(program)
            return

        # Separate function definitions from other statements
        functions: dict[str, FunctionDef] = {}
        other_statements: list[tuple[int, Statement]] = []

        for i, stmt in enumerate(program.statements):
            if isinstance(stmt, FunctionDef):
                functions[stmt.name] = stmt
            else:
                other_statements.append((i, stmt))

        # Get compilation order from call graph
        try:
            compilation_order = self._call_graph.topological_sort()
        except Exception:
            # Cycles detected - fall back to original order with graceful handling
            compilation_order = list(functions.keys())

        # Track which functions we've generated
        generated_functions: set[str] = set()

        # Generate non-function statements first (imports, top-level code)
        # that appear before the first function
        first_func_idx = min(
            (i for i, stmt in enumerate(program.statements) if isinstance(stmt, FunctionDef)),
            default=len(program.statements)
        )

        for idx, stmt in other_statements:
            if idx < first_func_idx:
                self.visit(stmt)
                if isinstance(stmt, (ClassDef, SceneDef)):
                    self._emit("")

        # Generate functions in compilation order (callees first)
        for func_name in compilation_order:
            if func_name in functions and func_name not in generated_functions:
                self.visit(functions[func_name])
                self._emit("")
                generated_functions.add(func_name)

        # Generate any remaining functions not in the call graph
        for func_name, func in functions.items():
            if func_name not in generated_functions:
                self.visit(func)
                self._emit("")
                generated_functions.add(func_name)

        # Generate remaining non-function statements (appear after functions)
        for idx, stmt in other_statements:
            if idx >= first_func_idx:
                self.visit(stmt)
                if isinstance(stmt, (ClassDef, SceneDef)):
                    self._emit("")

    def _is_jit_safe_by_purity(self, func_name: str) -> tuple[bool, str]:
        """
        Determine if a function is JIT-safe based on purity analysis.

        Uses purity analysis results for smarter JIT decisions:
        - PURE functions with no I/O -> definitely JIT-safe
        - IMPURE_IO functions -> don't JIT (I/O side effects)
        - IMPURE_MANIM functions -> don't JIT (Manim operations)
        - IMPURE_MUTATION -> generally okay for JIT (local mutations allowed)

        Args:
            func_name: Name of the function to check

        Returns:
            Tuple of (is_jit_safe, reason_string)
        """
        if func_name not in self._purity_info:
            # No purity info available - return unknown
            return True, "no purity analysis available"

        purity = self._purity_info[func_name]

        # Import Purity enum only when needed (avoid circular imports)
        from mathviz.compiler.purity_analyzer import Purity

        if purity.purity == Purity.PURE:
            return True, "function is pure"

        if purity.purity == Purity.IMPURE_IO:
            # I/O operations are not JIT-compatible
            io_ops = ", ".join(purity.io_calls[:3])
            if len(purity.io_calls) > 3:
                io_ops += "..."
            return False, f"function has I/O operations: {io_ops}"

        if purity.purity == Purity.IMPURE_MANIM:
            # Manim operations are not JIT-compatible
            manim_ops = ", ".join(purity.manim_calls[:3])
            if len(purity.manim_calls) > 3:
                manim_ops += "..."
            return False, f"function has Manim operations: {manim_ops}"

        if purity.purity == Purity.IMPURE_MUTATION:
            # Mutations to local or parameter arrays are okay for JIT
            # Global mutations might be okay depending on usage
            if purity.writes_globals:
                return True, "function mutates globals (may be JIT-compatible)"
            return True, "function has local mutations (JIT-compatible)"

        return True, "purity check passed"

    def _get_loop_analysis(self, loop_index: int) -> LoopAnalysis | None:
        """
        Get the loop analysis for the current function's loop at the given index.

        Args:
            loop_index: Zero-based index of the for-loop within the current function

        Returns:
            LoopAnalysis if available, None otherwise
        """
        if self._current_function is None:
            return None

        if self._current_function not in self._parallel_info:
            return None

        analyses = self._parallel_info[self._current_function]
        if loop_index < len(analyses):
            return analyses[loop_index]

        return None

    def _analyze_imports(self, program: Program) -> None:
        """Analyze the AST to determine what imports are needed."""
        parent = self

        class ImportAnalyzer(BaseASTVisitor):
            def __init__(self) -> None:
                self.parent = parent

            def visit_binary_expression(self, node: BinaryExpression) -> None:
                if node.operator in {
                    BinaryOperator.ELEMENT_OF,
                    BinaryOperator.NOT_ELEMENT_OF,
                    BinaryOperator.SUBSET,
                    BinaryOperator.SUPERSET,
                    BinaryOperator.PROPER_SUBSET,
                    BinaryOperator.PROPER_SUPERSET,
                    BinaryOperator.UNION,
                    BinaryOperator.INTERSECTION,
                    BinaryOperator.SET_DIFF,
                }:
                    self.parent._needs_runtime_import = True
                super().visit_binary_expression(node)

            def visit_function_def(self, node: FunctionDef) -> None:
                # Check if function has explicit JIT decorator OR is auto-JIT compatible
                will_jit = False
                if node.jit_options.mode != JitMode.NONE:
                    will_jit = True
                elif self.parent.optimize:
                    # Use purity-based JIT decision if available
                    if node.name in self.parent._purity_info:
                        is_safe, _ = self.parent._is_jit_safe_by_purity(node.name)
                        if is_safe:
                            # Also check basic compatibility
                            is_compatible, _ = self.parent._jit_analyzer.analyze(node)
                            if is_compatible:
                                will_jit = True
                    else:
                        # Fall back to basic compatibility check
                        is_compatible, _ = self.parent._jit_analyzer.analyze(node)
                        if is_compatible:
                            will_jit = True

                if will_jit:
                    self.parent._needs_numba_import = True
                    self.parent._needs_numpy_import = True  # NumPy for arrays in JIT

                # Check for parallelizable loops (needs prange)
                if node.name in self.parent._parallel_info:
                    for loop_analysis in self.parent._parallel_info[node.name]:
                        if loop_analysis.can_use_prange:
                            self.parent._needs_prange = True
                            self.parent._needs_numba_import = True
                            break

                super().visit_function_def(node)

            def visit_list_literal(self, node: ListLiteral) -> None:
                # If optimize mode, we'll use numpy arrays
                if self.parent.optimize and node.elements:
                    # Check if all elements are numeric
                    all_numeric = all(
                        isinstance(e, (IntegerLiteral, FloatLiteral))
                        for e in node.elements
                    )
                    if all_numeric:
                        self.parent._needs_numpy_import = True
                super().visit_list_literal(node)

            def visit_scene_def(self, node: SceneDef) -> None:
                self.parent._needs_manim_import = True
                self.parent._has_scene = True
                super().visit_scene_def(node)

            def visit_play_statement(self, node: PlayStatement) -> None:
                self.parent._needs_manim_import = True
                super().visit_play_statement(node)

            def visit_wait_statement(self, node: WaitStatement) -> None:
                self.parent._needs_manim_import = True
                super().visit_wait_statement(node)

            def visit_call_expression(self, node: CallExpression) -> None:
                # Detect I/O function usage and numpy functions
                if isinstance(node.callee, Identifier):
                    if node.callee.name in BUILTIN_IO_FUNCTIONS:
                        self.parent._needs_io_runtime = True
                    # Math functions that map to numpy
                    if node.callee.name in NUMPY_FUNCTION_MAP:
                        self.parent._needs_numpy_import = True
                # Detect iterator method usage
                if isinstance(node.callee, MemberAccess):
                    if node.callee.member in ITERATOR_METHOD_MAP:
                        self.parent._needs_iterator_runtime = True
                super().visit_call_expression(node)

            def visit_unwrap_expression(self, node: UnwrapExpression) -> None:
                # Unwrap requires the runtime helpers
                self.parent._needs_io_runtime = True
                super().visit_unwrap_expression(node)

            def visit_some_expression(self, node: SomeExpression) -> None:
                super().visit_some_expression(node)

            def visit_ok_expression(self, node: OkExpression) -> None:
                super().visit_ok_expression(node)

            def visit_err_expression(self, node: ErrExpression) -> None:
                super().visit_err_expression(node)

            def visit_tuple_literal(self, node: TupleLiteral) -> None:
                for elem in node.elements:
                    self.visit(elem)

            # OOP construct visitors
            def visit_struct_def(self, node: StructDef) -> None:
                self.parent._needs_dataclass_import = True
                self.parent._struct_types.add(node.name)
                super().visit_struct_def(node)

            def visit_trait_def(self, node: TraitDef) -> None:
                self.parent._needs_abc_import = True
                super().visit_trait_def(node)

            def visit_enum_def(self, node: EnumDef) -> None:
                # Check if enum has associated data
                has_data = any(variant.fields for variant in node.variants)
                if has_data:
                    self.parent._needs_dataclass_import = True
                else:
                    self.parent._needs_enum_import = True
                    # Track simple enums (those using Python Enum)
                    self.parent._simple_enums.add(node.name)
                # Track enum variants for correct codegen
                self.parent._enum_variants[node.name] = {v.name for v in node.variants}
                # Track which variants have data
                for variant in node.variants:
                    if variant.fields:
                        self.parent._enum_variants_with_data.add((node.name, variant.name))
                super().visit_enum_def(node)

            def visit_impl_block(self, node: ImplBlock) -> None:
                # Track methods for call conversion
                for method in node.methods:
                    if method.has_self:
                        # Instance methods: obj.method(args) -> Type_method(obj, args)
                        self.parent._instance_methods.add((node.target_type, method.name))
                    else:
                        # Static methods: Type.method(args) -> Type_method(args)
                        self.parent._static_methods.add((node.target_type, method.name))
                super().visit_impl_block(node)

            def visit_struct_literal(self, node: StructLiteral) -> None:
                # Check for struct spread syntax which needs dataclasses.replace
                if node.spread:
                    self.parent._needs_dataclass_replace = True
                super().visit_struct_literal(node)

        analyzer = ImportAnalyzer()
        analyzer.visit(program)

        # Check for main function
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef) and stmt.name == "main":
                self._has_main = True
                break

    def _emit(self, text: str) -> None:
        """Emit a line of code with current indentation."""
        if text:
            indent = " " * (self._indent_level * self.indent_size)
            self._output.append(f"{indent}{text}")
        else:
            self._output.append("")

    def _emit_raw(self, text: str) -> None:
        """Emit text without adding newline or indentation."""
        if self._output:
            self._output[-1] += text
        else:
            self._output.append(text)

    def _indent(self) -> None:
        """Increase indentation level."""
        self._indent_level += 1

    def _dedent(self) -> None:
        """Decrease indentation level."""
        self._indent_level = max(0, self._indent_level - 1)

    # -------------------------------------------------------------------------
    # Program and Block
    # -------------------------------------------------------------------------

    def visit_program(self, node: Program) -> None:
        """Generate code for the entire program."""
        for stmt in node.statements:
            self.visit(stmt)
            # Add blank line between top-level definitions
            if isinstance(stmt, (FunctionDef, ClassDef, SceneDef)):
                self._emit("")

    def visit_block(self, node: Block) -> str:
        """Generate code for a block of statements."""
        if not node.statements:
            self._emit("pass")
            return ""

        for stmt in node.statements:
            self.visit(stmt)

        return ""

    # -------------------------------------------------------------------------
    # Type Annotations
    # -------------------------------------------------------------------------

    def visit_simple_type(self, node: SimpleType) -> str:
        """Generate Python type annotation for a simple type."""
        return TYPE_TO_PYTHON.get(node.name, node.name)

    def visit_generic_type(self, node: GenericType) -> str:
        """Generate Python type annotation for a generic type."""
        base = TYPE_TO_PYTHON.get(node.base, node.base)
        args = ", ".join(self._generate_type(arg) for arg in node.type_args)
        return f"{base}[{args}]"

    def visit_function_type(self, node: FunctionType) -> str:
        """Generate Python type annotation for a function type."""
        params = ", ".join(self._generate_type(p) for p in node.param_types)
        ret = self._generate_type(node.return_type)
        return f"Callable[[{params}], {ret}]"

    def _generate_type(self, type_ann: TypeAnnotation) -> str:
        """Generate Python type string from a type annotation."""
        if isinstance(type_ann, SimpleType):
            return self.visit_simple_type(type_ann)
        elif isinstance(type_ann, GenericType):
            return self.visit_generic_type(type_ann)
        elif isinstance(type_ann, FunctionType):
            return self.visit_function_type(type_ann)
        return "Any"

    # -------------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------------

    def visit_identifier(self, node: Identifier) -> str:
        """Generate code for an identifier."""
        return node.name

    def visit_integer_literal(self, node: IntegerLiteral) -> str:
        """Generate code for an integer literal."""
        return str(node.value)

    def visit_float_literal(self, node: FloatLiteral) -> str:
        """Generate code for a float literal."""
        if node.value == float("inf"):
            return "float('inf')"
        if node.value == float("-inf"):
            return "float('-inf')"
        return repr(node.value)

    def visit_string_literal(self, node: StringLiteral) -> str:
        """Generate code for a string literal."""
        return repr(node.value)

    def visit_fstring(self, node: FString) -> str:
        """
        Generate code for an f-string (formatted string literal).

        Transforms MathViz f-strings into Python f-strings, handling:
        - Literal text parts
        - Expression interpolations with optional format specifiers
        """
        parts: list[str] = []

        for part in node.parts:
            if isinstance(part, FStringLiteral):
                # Escape special characters for f-string context
                escaped = (
                    part.value
                    .replace("\\", "\\\\")
                    .replace("{", "{{")
                    .replace("}", "}}")
                    .replace('"', '\\"')
                )
                parts.append(escaped)
            elif isinstance(part, FStringExpression):
                expr_code = self._generate_expr(part.expression)
                if part.format_spec:
                    parts.append(f"{{{expr_code}:{part.format_spec}}}")
                else:
                    parts.append(f"{{{expr_code}}}")

        return 'f"' + "".join(parts) + '"'

    def visit_boolean_literal(self, node: BooleanLiteral) -> str:
        """Generate code for a boolean literal."""
        return "True" if node.value else "False"

    def visit_none_literal(self, node: NoneLiteral) -> str:
        """Generate code for None literal."""
        return "None"

    def visit_list_literal(self, node: ListLiteral) -> str:
        """Generate code for a list literal, using numpy arrays when appropriate."""
        elements = ", ".join(self._generate_expr(e) for e in node.elements)

        # Use numpy array if in JIT function or if all elements are numeric
        if self.optimize and self._in_jit_function and node.elements:
            # Determine dtype from elements
            has_float = any(isinstance(e, FloatLiteral) for e in node.elements)
            all_numeric = all(
                isinstance(e, (IntegerLiteral, FloatLiteral))
                for e in node.elements
            )
            if all_numeric:
                dtype = "np.float64" if has_float else "np.int64"
                return f"np.array([{elements}], dtype={dtype})"

        return f"[{elements}]"

    def visit_set_literal(self, node: SetLiteral) -> str:
        """Generate code for a set literal."""
        if not node.elements:
            return "set()"
        elements = ", ".join(self._generate_expr(e) for e in node.elements)
        return f"{{{elements}}}"

    def visit_dict_literal(self, node: DictLiteral) -> str:
        """Generate code for a dictionary literal."""
        pairs = ", ".join(
            f"{self._generate_expr(k)}: {self._generate_expr(v)}"
            for k, v in node.pairs
        )
        return f"{{{pairs}}}"

    def visit_binary_expression(self, node: BinaryExpression) -> str:
        """Generate code for a binary expression."""
        left = self._generate_expr(node.left)
        right = self._generate_expr(node.right)

        # Handle mathematical set operators with runtime functions
        if node.operator == BinaryOperator.ELEMENT_OF:
            return f"is_element({left}, {right})"
        if node.operator == BinaryOperator.NOT_ELEMENT_OF:
            return f"is_not_element({left}, {right})"
        if node.operator == BinaryOperator.SUBSET:
            return f"is_subset({left}, {right})"
        if node.operator == BinaryOperator.SUPERSET:
            return f"is_superset({left}, {right})"
        if node.operator == BinaryOperator.PROPER_SUBSET:
            return f"({left} < {right})"  # Python set proper subset
        if node.operator == BinaryOperator.PROPER_SUPERSET:
            return f"({left} > {right})"  # Python set proper superset
        if node.operator == BinaryOperator.UNION:
            return f"set_union({left}, {right})"
        if node.operator == BinaryOperator.INTERSECTION:
            return f"set_intersection({left}, {right})"
        if node.operator == BinaryOperator.SET_DIFF:
            return f"set_difference({left}, {right})"

        # Standard Python operators
        op = BINARY_OP_TO_PYTHON.get(node.operator)
        if op:
            return f"({left} {op} {right})"

        raise ValueError(f"Unknown binary operator: {node.operator}")

    def visit_unary_expression(self, node: UnaryExpression) -> str:
        """Generate code for a unary expression."""
        operand = self._generate_expr(node.operand)
        op = UNARY_OP_TO_PYTHON[node.operator]
        if node.operator == UnaryOperator.NOT:
            return f"(not {operand})"
        return f"({op}{operand})"

    def visit_call_expression(self, node: CallExpression) -> str:
        """Generate code for a function call, converting to numpy when appropriate."""
        # Check for iterator method calls (e.g., list.map(|x| x * 2))
        if isinstance(node.callee, MemberAccess):
            method_name = node.callee.member
            if method_name in ITERATOR_METHOD_MAP:
                return self._generate_iterator_method_call(node)

        # Generate positional arguments
        positional_args = [self._generate_expr(arg) for arg in node.arguments]

        # Generate keyword arguments
        keyword_args = [
            f"{kwarg.name}={self._generate_expr(kwarg.value)}"
            for kwarg in node.keyword_arguments
        ]

        # Combine all arguments
        all_args = positional_args + keyword_args
        args = ", ".join(all_args)

        # Check if we should convert to numpy function or I/O function
        if isinstance(node.callee, Identifier):
            func_name = node.callee.name

            # I/O functions
            if func_name in BUILTIN_IO_FUNCTIONS:
                self._needs_io_runtime = True
                return f"{BUILTIN_IO_FUNCTIONS[func_name]}({args})"

            # Convert to numpy if in JIT context or if it's a math function
            if self._in_jit_function or self.optimize:
                if func_name in NUMPY_FUNCTION_MAP:
                    self._needs_numpy_import = True
                    return f"{NUMPY_FUNCTION_MAP[func_name]}({args})"

        # Check for static method calls: Type.method() -> Type_method()
        if isinstance(node.callee, MemberAccess):
            if isinstance(node.callee.object, Identifier):
                type_name = node.callee.object.name
                method_name = node.callee.member
                if (type_name, method_name) in self._static_methods:
                    return f"{type_name}_{method_name}({args})"

            # Check for instance method calls: obj.method(args) -> Type_method(obj, args)
            method_name = node.callee.member
            for struct_type, m_name in self._instance_methods:
                if m_name == method_name:
                    # Found matching instance method - convert call
                    obj = self._generate_expr(node.callee.object)
                    if args:
                        return f"{struct_type}_{method_name}({obj}, {args})"
                    else:
                        return f"{struct_type}_{method_name}({obj})"

        callee = self._generate_expr(node.callee)
        return f"{callee}({args})"

    def _generate_iterator_method_call(self, node: CallExpression) -> str:
        """
        Generate code for iterator method calls.

        Transforms method calls like:
            list.map(|x| x * 2) -> iter_map(list, lambda x: x * 2)
            list.filter(|x| x > 0) -> iter_filter(list, lambda x: x > 0)
            list.reduce(0, |acc, x| acc + x) -> iter_reduce(list, 0, lambda acc, x: acc + x)
        """
        self._needs_iterator_runtime = True

        assert isinstance(node.callee, MemberAccess)
        method_name = node.callee.member
        obj = self._generate_expr(node.callee.object)

        runtime_func, takes_lambda, extra_args = ITERATOR_METHOD_MAP[method_name]

        # Generate arguments
        args: list[str] = [obj]  # First argument is always the iterable

        if method_name in ("reduce", "fold"):
            # reduce/fold has (initial, func) as arguments
            if len(node.arguments) >= 2:
                initial = self._generate_expr(node.arguments[0])
                func = self._generate_expr(node.arguments[1])
                args.append(initial)
                args.append(func)
        elif takes_lambda and len(node.arguments) >= 1:
            # Methods like map, filter take a lambda as first argument
            func = self._generate_expr(node.arguments[0])
            args.append(func)
        elif extra_args > 0:
            # Methods like take, skip, nth take additional arguments
            for arg in node.arguments[:extra_args]:
                args.append(self._generate_expr(arg))
        elif extra_args == -1:
            # Variadic methods like chain
            for arg in node.arguments:
                args.append(self._generate_expr(arg))

        return f"{runtime_func}({', '.join(args)})"

    def visit_member_access(self, node: MemberAccess) -> str:
        """Generate code for member access."""
        obj = self._generate_expr(node.object)
        return f"{obj}.{node.member}"

    def visit_index_expression(self, node: IndexExpression) -> str:
        """Generate code for index/subscript access."""
        obj = self._generate_expr(node.object)
        index = self._generate_expr(node.index)
        return f"{obj}[{index}]"

    def visit_conditional_expression(self, node: ConditionalExpression) -> str:
        """Generate code for a ternary conditional."""
        then_expr = self._generate_expr(node.then_expr)
        condition = self._generate_expr(node.condition)
        else_expr = self._generate_expr(node.else_expr)
        return f"({then_expr} if {condition} else {else_expr})"

    def visit_lambda_expression(self, node: LambdaExpression) -> str:
        """Generate code for a lambda expression."""
        params = ", ".join(p.name for p in node.parameters)
        if isinstance(node.body, Block):
            # Multi-statement lambda needs to be a regular function
            # For simplicity, just use the first return statement
            raise NotImplementedError(
                "Block-bodied lambdas require function extraction"
            )
        body = self._generate_expr(node.body)
        return f"lambda {params}: {body}"

    def visit_range_expression(self, node: RangeExpression) -> str:
        """Generate code for a range expression."""
        start = self._generate_expr(node.start)
        end = self._generate_expr(node.end)
        if node.inclusive:
            return f"range({start}, {end} + 1)"
        return f"range({start}, {end})"

    def visit_list_comprehension(self, node: ListComprehension) -> str:
        """
        Generate code for a list comprehension.

        Examples:
            [x^2 for x in 0..10] -> [x ** 2 for x in range(0, 10)]
            [x for x in items if x > 0] -> [x for x in items if x > 0]
            [(x, y) for x in 0..3 for y in 0..3] -> [(x, y) for x in range(0, 3) for y in range(0, 3)]
        """
        element = self._generate_expr(node.element)
        clauses = self._generate_comprehension_clauses(node.clauses)
        return f"[{element} {clauses}]"

    def visit_set_comprehension(self, node: SetComprehension) -> str:
        """
        Generate code for a set comprehension.

        Examples:
            {x % 10 for x in items} -> {x % 10 for x in items}
        """
        element = self._generate_expr(node.element)
        clauses = self._generate_comprehension_clauses(node.clauses)
        return f"{{{element} {clauses}}}"

    def visit_dict_comprehension(self, node: DictComprehension) -> str:
        """
        Generate code for a dict comprehension.

        Examples:
            {k: v for (k, v) in items} -> {k: v for (k, v) in items}
            {x: x^2 for x in 0..10} -> {x: x ** 2 for x in range(0, 10)}
        """
        key = self._generate_expr(node.key)
        value = self._generate_expr(node.value)
        clauses = self._generate_comprehension_clauses(node.clauses)
        return f"{{{key}: {value} {clauses}}}"

    def _generate_comprehension_clauses(self, clauses: tuple[ComprehensionClause, ...]) -> str:
        """
        Generate Python code for comprehension clauses.

        Each clause becomes 'for var in iterable [if condition]'.
        """
        parts: list[str] = []
        for clause in clauses:
            iterable = self._generate_expr(clause.iterable)
            clause_str = f"for {clause.variable} in {iterable}"
            if clause.condition:
                condition = self._generate_expr(clause.condition)
                clause_str += f" if {condition}"
            parts.append(clause_str)
        return " ".join(parts)

    def visit_pipe_lambda(self, node: PipeLambda) -> str:
        """
        Generate code for a pipe-style lambda expression.

        Examples:
            |x| x * 2 -> lambda x: x * 2
            |x, y| x + y -> lambda x, y: x + y
        """
        params = ", ".join(node.parameters)
        body = self._generate_expr(node.body)
        return f"lambda {params}: {body}"

    def visit_tuple_literal(self, node: TupleLiteral) -> str:
        """Generate code for a tuple literal."""
        elements = ", ".join(self._generate_expr(e) for e in node.elements)
        if len(node.elements) == 0:
            return "()"
        if len(node.elements) == 1:
            return f"({elements},)"
        return f"({elements})"

    def visit_some_expression(self, node: SomeExpression) -> str:
        """
        Generate code for a Some(value) expression.

        In Python, Some(value) simply becomes the value itself,
        since Optional is represented as Union[T, None].
        """
        return self._generate_expr(node.value)

    def visit_ok_expression(self, node: OkExpression) -> str:
        """
        Generate code for an Ok(value) expression.

        Uses a tuple (True, value) to represent Ok.
        """
        value = self._generate_expr(node.value)
        return f"(True, {value})"

    def visit_err_expression(self, node: ErrExpression) -> str:
        """
        Generate code for an Err(error) expression.

        Uses a tuple (False, error) to represent Err.
        """
        value = self._generate_expr(node.value)
        return f"(False, {value})"

    def visit_unwrap_expression(self, node: UnwrapExpression) -> str:
        """
        Generate code for an unwrap expression (value?).

        For Optional, uses a helper that raises if None.
        For Result, uses a helper that raises on Err.
        """
        operand = self._generate_expr(node.operand)
        # Generate inline unwrap - for Result types, check the first element
        # For Optional, just use the value directly (None check happens at runtime)
        # This is a simplified implementation; in production, we'd emit runtime helpers
        return f"_mviz_unwrap({operand})"

    def visit_match_expression(self, node: MatchExpression) -> str:
        """
        Generate code for a match expression.

        Match expressions are translated to a series of if-elif-else statements
        wrapped in a lambda that's immediately called, to allow use as an expression.

        Example:
            match value {
                0 -> "zero"
                n where n > 0 -> "positive"
                _ -> "negative"
            }

        Becomes:
            (lambda _match_subject: (
                "zero" if _match_subject == 0 else
                ("positive" if (_match_subject > 0) else
                "negative")
            ))(value)
        """
        # Generate a unique variable name for the match subject
        subject = self._generate_expr(node.subject)

        # Simple cases can use nested ternary expressions
        # More complex cases (with blocks) need IIFE pattern
        has_blocks = any(isinstance(arm.body, Block) for arm in node.arms)

        if has_blocks:
            # Use immediately-invoked function for block bodies
            return self._generate_match_as_iife(node, subject)
        else:
            # Use nested ternary expression for simple expression bodies
            return self._generate_match_as_ternary(node, subject)

    def _generate_match_as_ternary(self, node: MatchExpression, subject: str) -> str:
        """Generate match as nested ternary expressions."""
        # Start building the nested ternary from the last arm
        arms = list(node.arms)

        def build_ternary(arms_remaining: list[MatchArm], subject_var: str) -> str:
            if not arms_remaining:
                # Fallback - should not happen with exhaustive match
                return "None"

            arm = arms_remaining[0]
            rest = arms_remaining[1:]

            # Generate condition for this arm
            condition = self._generate_pattern_condition(arm.pattern, subject_var)

            # Add guard if present
            if arm.guard:
                guard_code = self._generate_expr(arm.guard)
                if condition != "True":
                    condition = f"({condition} and {guard_code})"
                else:
                    condition = guard_code

            # Generate body with pattern bindings
            bindings = self._generate_pattern_bindings(arm.pattern, subject_var)
            if bindings:
                # Need to use a lambda to create scope for bindings
                body_expr = self._generate_expr(arm.body)
                body_with_bindings = f"(lambda {', '.join(bindings.keys())}: {body_expr})({', '.join(bindings.values())})"
            else:
                body_with_bindings = self._generate_expr(arm.body)

            # Build ternary
            if not rest:
                # Last arm (or only arm)
                if condition == "True":
                    return body_with_bindings
                return f"({body_with_bindings} if {condition} else None)"
            else:
                else_branch = build_ternary(rest, subject_var)
                if condition == "True":
                    return body_with_bindings
                return f"({body_with_bindings} if {condition} else {else_branch})"

        # Wrap the whole thing in a lambda to evaluate subject once
        ternary = build_ternary(arms, "_match_subj")
        return f"(lambda _match_subj: {ternary})({subject})"

    def _generate_match_as_iife(self, node: MatchExpression, subject: str) -> str:
        """
        Generate match as immediately-invoked function expression (IIFE).

        This is used when arms have block bodies that can't be expressed as ternaries.
        """
        # We'll generate inline helper that returns the matched value
        lines = []
        lines.append(f"(lambda _match_subj: (")

        # Generate if-elif chain
        first = True
        for arm in node.arms:
            condition = self._generate_pattern_condition(arm.pattern, "_match_subj")

            if arm.guard:
                guard_code = self._generate_expr(arm.guard)
                if condition != "True":
                    condition = f"({condition} and {guard_code})"
                else:
                    condition = guard_code

            bindings = self._generate_pattern_bindings(arm.pattern, "_match_subj")

            if isinstance(arm.body, Block):
                # Block body - extract last expression or return None
                body_expr = "None"
                # For now, simplify: if it's a block, just use None
                # A full implementation would need to analyze the block
            else:
                body_expr = self._generate_expr(arm.body)

            if bindings:
                body_expr = f"(lambda {', '.join(bindings.keys())}: {body_expr})({', '.join(bindings.values())})"

            if condition == "True":
                lines.append(f"    {body_expr}")
            elif first:
                lines.append(f"    {body_expr} if {condition}")
                first = False
            else:
                lines.append(f"    else {body_expr} if {condition}")

        lines.append(f"    else None")
        lines.append(f"))({subject})")

        return " ".join(lines)

    def _generate_pattern_condition(self, pattern: Pattern, subject_var: str) -> str:
        """Generate a condition that checks if the subject matches the pattern."""
        if isinstance(pattern, LiteralPattern):
            literal = self._generate_expr(pattern.value)
            return f"{subject_var} == {literal}"

        elif isinstance(pattern, IdentifierPattern):
            # Identifier patterns always match (they just bind)
            return "True"

        elif isinstance(pattern, TuplePattern):
            # Check length and recursively check elements
            conditions = [f"isinstance({subject_var}, tuple)"]
            conditions.append(f"len({subject_var}) == {len(pattern.elements)}")
            for i, elem in enumerate(pattern.elements):
                elem_cond = self._generate_pattern_condition(elem, f"{subject_var}[{i}]")
                if elem_cond != "True":
                    conditions.append(elem_cond)
            if len(conditions) == 1:
                return conditions[0]
            return " and ".join(f"({c})" for c in conditions)

        elif isinstance(pattern, ConstructorPattern):
            # Handle Some/Ok/Err patterns
            if pattern.name == "Some":
                # Some matches if not None
                conditions = [f"{subject_var} is not None"]
                for i, arg in enumerate(pattern.args):
                    arg_cond = self._generate_pattern_condition(arg, subject_var)
                    if arg_cond != "True":
                        conditions.append(arg_cond)
                return " and ".join(f"({c})" for c in conditions)

            elif pattern.name == "Ok":
                # Ok matches if it's a tuple with True as first element
                conditions = [
                    f"isinstance({subject_var}, tuple)",
                    f"len({subject_var}) == 2",
                    f"{subject_var}[0] is True"
                ]
                for i, arg in enumerate(pattern.args):
                    arg_cond = self._generate_pattern_condition(arg, f"{subject_var}[1]")
                    if arg_cond != "True":
                        conditions.append(arg_cond)
                return " and ".join(f"({c})" for c in conditions)

            elif pattern.name == "Err":
                # Err matches if it's a tuple with False as first element
                conditions = [
                    f"isinstance({subject_var}, tuple)",
                    f"len({subject_var}) == 2",
                    f"{subject_var}[0] is False"
                ]
                for i, arg in enumerate(pattern.args):
                    arg_cond = self._generate_pattern_condition(arg, f"{subject_var}[1]")
                    if arg_cond != "True":
                        conditions.append(arg_cond)
                return " and ".join(f"({c})" for c in conditions)

            else:
                # Generic constructor pattern
                conditions = [f"isinstance({subject_var}, {pattern.name})"]
                return " and ".join(conditions)

        elif isinstance(pattern, EnumPattern):
            # Enum pattern: check isinstance for the variant class
            # For dataclass-based enums: isinstance(subject, VariantName)
            # Handle reserved keywords like None -> _None
            variant_name = pattern.variant_name
            if variant_name == "None":
                variant_name = "_None"
            elif variant_name == "True":
                variant_name = "_True"
            elif variant_name == "False":
                variant_name = "_False"
            conditions = [f"isinstance({subject_var}, {variant_name})"]
            for i, binding in enumerate(pattern.bindings):
                binding_cond = self._generate_pattern_condition(binding, f"{subject_var}._{i}")
                if binding_cond != "True":
                    conditions.append(binding_cond)
            return " and ".join(f"({c})" for c in conditions)

        elif isinstance(pattern, RangePattern):
            # Range pattern: check if subject is within range
            start = self._generate_expr(pattern.start)
            end = self._generate_expr(pattern.end)
            if pattern.inclusive:
                return f"({start} <= {subject_var} <= {end})"
            else:
                return f"({start} <= {subject_var} < {end})"

        elif isinstance(pattern, OrPattern):
            # Or pattern: check if any sub-pattern matches
            sub_conditions = []
            for sub in pattern.patterns:
                sub_cond = self._generate_pattern_condition(sub, subject_var)
                sub_conditions.append(f"({sub_cond})")
            return " or ".join(sub_conditions)

        elif isinstance(pattern, BindingPattern):
            # Binding pattern: check inner pattern (binding happens in _generate_pattern_bindings)
            return self._generate_pattern_condition(pattern.pattern, subject_var)

        elif isinstance(pattern, ListPattern):
            # List pattern: check length and elements, handling RestPattern
            conditions = [f"isinstance({subject_var}, (list, tuple))"]

            # Count non-rest elements
            rest_index = None
            non_rest_count = 0
            for i, elem in enumerate(pattern.elements):
                if isinstance(elem, RestPattern):
                    rest_index = i
                else:
                    non_rest_count += 1

            if rest_index is None:
                # No rest pattern - exact length match
                conditions.append(f"len({subject_var}) == {len(pattern.elements)}")
                for i, elem in enumerate(pattern.elements):
                    elem_cond = self._generate_pattern_condition(elem, f"{subject_var}[{i}]")
                    if elem_cond != "True":
                        conditions.append(elem_cond)
            else:
                # Has rest pattern - minimum length match
                conditions.append(f"len({subject_var}) >= {non_rest_count}")
                # Check elements before rest
                for i in range(rest_index):
                    elem = pattern.elements[i]
                    elem_cond = self._generate_pattern_condition(elem, f"{subject_var}[{i}]")
                    if elem_cond != "True":
                        conditions.append(elem_cond)
                # Check elements after rest (from the end)
                after_rest = len(pattern.elements) - rest_index - 1
                for j in range(after_rest):
                    elem = pattern.elements[rest_index + 1 + j]
                    elem_cond = self._generate_pattern_condition(elem, f"{subject_var}[-{after_rest - j}]")
                    if elem_cond != "True":
                        conditions.append(elem_cond)

            if len(conditions) == 1:
                return conditions[0]
            return " and ".join(f"({c})" for c in conditions)

        elif isinstance(pattern, RestPattern):
            # Rest pattern always matches (it captures remaining elements)
            return "True"

        return "True"

    def _generate_pattern_bindings(self, pattern: Pattern, subject_var: str) -> dict[str, str]:
        """
        Generate variable bindings for a pattern.

        Returns a dict mapping variable names to the expressions that provide their values.
        """
        bindings: dict[str, str] = {}

        if isinstance(pattern, LiteralPattern):
            # Literal patterns don't bind
            pass

        elif isinstance(pattern, IdentifierPattern):
            if not pattern.is_wildcard:
                bindings[pattern.name] = subject_var

        elif isinstance(pattern, TuplePattern):
            for i, elem in enumerate(pattern.elements):
                elem_bindings = self._generate_pattern_bindings(elem, f"{subject_var}[{i}]")
                bindings.update(elem_bindings)

        elif isinstance(pattern, ConstructorPattern):
            if pattern.name == "Some":
                # Some(x) binds x to the value
                for i, arg in enumerate(pattern.args):
                    arg_bindings = self._generate_pattern_bindings(arg, subject_var)
                    bindings.update(arg_bindings)
            elif pattern.name in ("Ok", "Err"):
                # Ok(x) / Err(x) binds x to the inner value (second element of tuple)
                for i, arg in enumerate(pattern.args):
                    arg_bindings = self._generate_pattern_bindings(arg, f"{subject_var}[1]")
                    bindings.update(arg_bindings)
            else:
                # Generic constructor - assume args map to properties
                for i, arg in enumerate(pattern.args):
                    arg_bindings = self._generate_pattern_bindings(arg, f"{subject_var}._{i}")
                    bindings.update(arg_bindings)

        elif isinstance(pattern, EnumPattern):
            # Enum pattern bindings: access fields by position (_0, _1, etc.)
            for i, binding in enumerate(pattern.bindings):
                binding_bindings = self._generate_pattern_bindings(binding, f"{subject_var}._{i}")
                bindings.update(binding_bindings)

        elif isinstance(pattern, RangePattern):
            # Range patterns don't bind variables
            pass

        elif isinstance(pattern, OrPattern):
            # Or patterns: collect bindings from all sub-patterns
            # All alternatives must bind the same variables
            for sub in pattern.patterns:
                sub_bindings = self._generate_pattern_bindings(sub, subject_var)
                bindings.update(sub_bindings)

        elif isinstance(pattern, BindingPattern):
            # Binding pattern: bind the name to subject, plus inner bindings
            bindings[pattern.name] = subject_var
            inner_bindings = self._generate_pattern_bindings(pattern.pattern, subject_var)
            bindings.update(inner_bindings)

        elif isinstance(pattern, ListPattern):
            # List pattern: bind elements, handling RestPattern
            rest_index = None
            for i, elem in enumerate(pattern.elements):
                if isinstance(elem, RestPattern):
                    rest_index = i
                    break

            if rest_index is None:
                # No rest - simple indexing
                for i, elem in enumerate(pattern.elements):
                    elem_bindings = self._generate_pattern_bindings(elem, f"{subject_var}[{i}]")
                    bindings.update(elem_bindings)
            else:
                # Has rest pattern
                # Elements before rest
                for i in range(rest_index):
                    elem = pattern.elements[i]
                    elem_bindings = self._generate_pattern_bindings(elem, f"{subject_var}[{i}]")
                    bindings.update(elem_bindings)

                # Rest element
                rest_elem = pattern.elements[rest_index]
                if isinstance(rest_elem, RestPattern) and rest_elem.name:
                    after_rest = len(pattern.elements) - rest_index - 1
                    if after_rest == 0:
                        bindings[rest_elem.name] = f"{subject_var}[{rest_index}:]"
                    else:
                        bindings[rest_elem.name] = f"{subject_var}[{rest_index}:-{after_rest}]"

                # Elements after rest
                after_rest = len(pattern.elements) - rest_index - 1
                for j in range(after_rest):
                    elem = pattern.elements[rest_index + 1 + j]
                    elem_bindings = self._generate_pattern_bindings(elem, f"{subject_var}[-{after_rest - j}]")
                    bindings.update(elem_bindings)

        elif isinstance(pattern, RestPattern):
            # Rest pattern with name binds to remaining elements
            if pattern.name:
                bindings[pattern.name] = subject_var

        return bindings

    def _generate_expr(self, expr: Expression) -> str:
        """Generate Python code for an expression."""
        if isinstance(expr, Identifier):
            return self.visit_identifier(expr)
        elif isinstance(expr, IntegerLiteral):
            return self.visit_integer_literal(expr)
        elif isinstance(expr, FloatLiteral):
            return self.visit_float_literal(expr)
        elif isinstance(expr, StringLiteral):
            return self.visit_string_literal(expr)
        elif isinstance(expr, FString):
            return self.visit_fstring(expr)
        elif isinstance(expr, BooleanLiteral):
            return self.visit_boolean_literal(expr)
        elif isinstance(expr, NoneLiteral):
            return self.visit_none_literal(expr)
        elif isinstance(expr, ListLiteral):
            return self.visit_list_literal(expr)
        elif isinstance(expr, SetLiteral):
            return self.visit_set_literal(expr)
        elif isinstance(expr, DictLiteral):
            return self.visit_dict_literal(expr)
        elif isinstance(expr, BinaryExpression):
            return self.visit_binary_expression(expr)
        elif isinstance(expr, UnaryExpression):
            return self.visit_unary_expression(expr)
        elif isinstance(expr, CallExpression):
            return self.visit_call_expression(expr)
        elif isinstance(expr, MemberAccess):
            return self.visit_member_access(expr)
        elif isinstance(expr, IndexExpression):
            return self.visit_index_expression(expr)
        elif isinstance(expr, ConditionalExpression):
            return self.visit_conditional_expression(expr)
        elif isinstance(expr, LambdaExpression):
            return self.visit_lambda_expression(expr)
        elif isinstance(expr, RangeExpression):
            return self.visit_range_expression(expr)
        elif isinstance(expr, TupleLiteral):
            return self.visit_tuple_literal(expr)
        elif isinstance(expr, SomeExpression):
            return self.visit_some_expression(expr)
        elif isinstance(expr, OkExpression):
            return self.visit_ok_expression(expr)
        elif isinstance(expr, ErrExpression):
            return self.visit_err_expression(expr)
        elif isinstance(expr, UnwrapExpression):
            return self.visit_unwrap_expression(expr)
        elif isinstance(expr, MatchExpression):
            return self.visit_match_expression(expr)
        elif isinstance(expr, SelfExpression):
            return self.visit_self_expression(expr)
        elif isinstance(expr, EnumVariantAccess):
            return self.visit_enum_variant_access(expr)
        elif isinstance(expr, StructLiteral):
            return self.visit_struct_literal(expr)
        elif isinstance(expr, ListComprehension):
            return self.visit_list_comprehension(expr)
        elif isinstance(expr, SetComprehension):
            return self.visit_set_comprehension(expr)
        elif isinstance(expr, DictComprehension):
            return self.visit_dict_comprehension(expr)
        elif isinstance(expr, PipeLambda):
            return self.visit_pipe_lambda(expr)
        elif isinstance(expr, AwaitExpression):
            return self.visit_await_expression(expr)
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    # -------------------------------------------------------------------------
    # Statements
    # -------------------------------------------------------------------------

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        """Generate code for an expression statement."""
        self._emit(self._generate_expr(node.expression))

    def visit_let_statement(self, node: LetStatement) -> None:
        """Generate code for a variable declaration."""
        if node.type_annotation:
            type_hint = self._generate_type(node.type_annotation)
            if node.value:
                value = self._generate_expr(node.value)
                self._emit(f"{node.name}: {type_hint} = {value}")
            else:
                self._emit(f"{node.name}: {type_hint}")
        else:
            if node.value:
                value = self._generate_expr(node.value)
                self._emit(f"{node.name} = {value}")
            else:
                self._emit(f"{node.name} = None")

    def visit_destructuring_let_statement(self, node: "DestructuringLetStatement") -> None:
        """Generate code for a destructuring variable declaration."""
        names = ", ".join(node.names)
        value = self._generate_expr(node.value)
        self._emit(f"{names} = {value}")

    def visit_const_declaration(self, node: ConstDeclaration) -> None:
        """
        Generate code for a compile-time constant declaration.

        Constants are generated as module-level Python constants (uppercase by convention).
        The const_evaluator is used to compute values at compile time when possible.
        """
        from mathviz.compiler.const_evaluator import ConstEvaluator, ConstEvalError

        # Try to evaluate the constant at compile time
        try:
            evaluator = ConstEvaluator()
            const_value = evaluator.evaluate(node.value)
            # Emit the computed value directly
            self._emit(f"{node.name} = {repr(const_value)}")
        except ConstEvalError:
            # Fall back to runtime evaluation if compile-time eval fails
            value = self._generate_expr(node.value)
            if node.type_annotation:
                type_hint = self._generate_type(node.type_annotation)
                self._emit(f"{node.name}: {type_hint} = {value}")
            else:
                self._emit(f"{node.name} = {value}")

    def visit_assignment_statement(self, node: AssignmentStatement) -> None:
        """Generate code for an assignment."""
        target = self._generate_expr(node.target)
        value = self._generate_expr(node.value)
        self._emit(f"{target} = {value}")

    def visit_compound_assignment(self, node: CompoundAssignment) -> None:
        """Generate code for compound assignment."""
        target = self._generate_expr(node.target)
        value = self._generate_expr(node.value)
        op = COMPOUND_OP_TO_PYTHON[node.operator]
        self._emit(f"{target} {op} {value}")

    def visit_function_def(self, node: FunctionDef) -> None:
        """
        Generate code for a function definition with automatic Numba JIT optimization.

        Uses purity analysis (when available) for smarter JIT decisions:
        - PURE functions with no I/O -> definitely JIT-safe
        - IMPURE_IO functions -> don't JIT
        - IMPURE_MANIM functions -> don't JIT

        Also uses parallel analysis to determine if parallel=True should be set
        for functions with parallelizable loops.

        For generic functions, generates TypeVar declarations:
            fn identity<T>(x: T) -> T { return x }
        becomes:
            T = TypeVar('T')
            def identity(x: T) -> T:
                return x
        """
        # Handle generic type parameters for functions
        if node.type_params:
            self._needs_typing_import = True
            # Generate TypeVar declarations for new type parameters
            for type_param in node.type_params:
                if type_param.name not in self._type_vars_declared:
                    self._type_vars_declared.add(type_param.name)
                    if type_param.bounds:
                        bounds_str = ", ".join(type_param.bounds)
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}', bound={bounds_str})")
                    else:
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}')")

        jit_opts = node.jit_options

        # Track the current function for parallel info lookup
        prev_function = self._current_function
        self._current_function = node.name
        self._current_loop_index = 0

        # Emit complexity comment if verbose mode is enabled
        if self.verbose and node.name in self._complexity_info:
            complexity = self._complexity_info[node.name]
            self._emit(f"# Complexity: {complexity.complexity.value} - {complexity.explanation}")

        # Determine if we should apply JIT
        should_jit = False
        effective_jit_opts = jit_opts
        needs_parallel = False

        if jit_opts.mode != JitMode.NONE:
            # Explicit decorator - use it
            should_jit = True
        elif self.optimize and not self._in_scene:
            # Use purity-based JIT decision if available
            if node.name in self._purity_info:
                is_purity_safe, purity_reason = self._is_jit_safe_by_purity(node.name)
                if is_purity_safe:
                    # Also check basic type compatibility
                    is_compatible, _ = self._jit_analyzer.analyze(node)
                    if is_compatible:
                        should_jit = True
                # If purity check fails, don't JIT
            else:
                # Fall back to basic compatibility check (original behavior)
                is_compatible, _ = self._jit_analyzer.analyze(node)
                if is_compatible:
                    should_jit = True

            # Check if function has parallelizable loops
            if should_jit and node.name in self._parallel_info:
                for loop_analysis in self._parallel_info[node.name]:
                    if loop_analysis.can_use_prange and loop_analysis.needs_parallel_flag:
                        needs_parallel = True
                        break

            if should_jit:
                # Use default optimal settings for auto-JIT
                effective_jit_opts = JitOptions(
                    mode=JitMode.NJIT,  # Use njit for best performance
                    nopython=True,
                    nogil=False,
                    cache=True,         # Cache for faster subsequent runs
                    parallel=needs_parallel,  # Enable if we detect parallelizable loops with reductions
                    fastmath=False,     # Safe by default
                    boundscheck=False,
                )

        if should_jit:
            decorator = self._generate_jit_decorator(effective_jit_opts)
            self._emit(decorator)

        # Track if we're in a JIT function for numpy array generation
        was_in_jit = self._in_jit_function
        self._in_jit_function = should_jit

        # Build parameter list
        params: list[str] = []
        for param in node.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {self._generate_type(param.type_annotation)}"
            if param.default_value:
                param_str += f" = {self._generate_expr(param.default_value)}"
            params.append(param_str)

        params_str = ", ".join(params)

        # Return type annotation
        return_type = ""
        if node.return_type:
            return_type = f" -> {self._generate_type(node.return_type)}"

        self._emit(f"def {node.name}({params_str}){return_type}:")
        self._indent()
        if node.body:
            self.visit(node.body)
        else:
            self._emit("pass")
        self._dedent()

        # Restore context
        self._in_jit_function = was_in_jit
        self._current_function = prev_function
        self._current_loop_index = 0

    def visit_async_function_def(self, node: AsyncFunctionDef) -> None:
        """Generate code for an async function definition."""
        self._needs_asyncio_import = True

        # Build parameter list
        params: list[str] = []
        for param in node.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {self._generate_type(param.type_annotation)}"
            if param.default_value:
                param_str += f" = {self._generate_expr(param.default_value)}"
            params.append(param_str)

        params_str = ", ".join(params)

        # Return type annotation
        return_type = ""
        if node.return_type:
            return_type = f" -> {self._generate_type(node.return_type)}"

        self._emit(f"async def {node.name}({params_str}){return_type}:")
        self._indent()
        if node.body:
            self.visit(node.body)
        else:
            self._emit("pass")
        self._dedent()

    def visit_async_for_statement(self, node: AsyncForStatement) -> None:
        """Generate code for an async for loop."""
        iterable = self._generate_expr(node.iterable)
        self._emit(f"async for {node.variable} in {iterable}:")
        self._indent()
        self.visit(node.body)
        self._dedent()

    def visit_await_expression(self, node: AwaitExpression) -> str:
        """Generate code for an await expression."""
        expr = self._generate_expr(node.expression)
        return f"(await {expr})"

    def _generate_jit_decorator(self, opts: JitOptions) -> str:
        """Generate a Numba JIT decorator string with all options."""
        args = []

        if opts.mode == JitMode.JIT:
            decorator_name = "jit"
            if opts.nopython:
                args.append("nopython=True")
        elif opts.mode == JitMode.NJIT:
            decorator_name = "njit"
            # njit already implies nopython=True
        elif opts.mode == JitMode.VECTORIZE:
            decorator_name = "vectorize"
        else:
            decorator_name = "jit"

        # Add optional arguments
        if opts.nogil:
            args.append("nogil=True")
        if opts.cache:
            args.append("cache=True")
        if opts.parallel:
            args.append("parallel=True")
        if opts.fastmath:
            args.append("fastmath=True")
        if opts.boundscheck:
            args.append("boundscheck=True")
        if opts.inline != "never":
            args.append(f"inline='{opts.inline}'")

        if args:
            return f"@{decorator_name}({', '.join(args)})"
        return f"@{decorator_name}"

    def visit_class_def(self, node: ClassDef) -> None:
        """Generate code for a class definition."""
        bases = ", ".join(node.base_classes) if node.base_classes else ""
        if bases:
            self._emit(f"class {node.name}({bases}):")
        else:
            self._emit(f"class {node.name}:")

        self._indent()
        if node.body:
            self.visit(node.body)
        else:
            self._emit("pass")
        self._dedent()

    def visit_scene_def(self, node: SceneDef) -> None:
        """Generate code for a Manim scene definition."""
        self._emit(f"class {node.name}(Scene):")
        self._indent()
        self._in_scene = True
        if node.body:
            self.visit(node.body)
        else:
            self._emit("pass")
        self._in_scene = False
        self._dedent()

    def visit_if_statement(self, node: IfStatement) -> None:
        """Generate code for an if statement."""
        condition = self._generate_expr(node.condition)
        self._emit(f"if {condition}:")
        self._indent()
        self.visit(node.then_block)
        self._dedent()

        for elif_cond, elif_block in node.elif_clauses:
            elif_cond_str = self._generate_expr(elif_cond)
            self._emit(f"elif {elif_cond_str}:")
            self._indent()
            self.visit(elif_block)
            self._dedent()

        if node.else_block:
            self._emit("else:")
            self._indent()
            self.visit(node.else_block)
            self._dedent()

    def visit_for_statement(self, node: ForStatement) -> None:
        """
        Generate code for a for loop, using prange when parallelizable.

        When parallel analysis data is available and the loop is marked as
        parallelizable, this method will:
        - Replace range() with prange() for parallel execution
        - Add appropriate comments about the parallelization

        The prange function from Numba enables automatic parallelization
        of loops when used within a function decorated with @njit(parallel=True).
        """
        # Check if this loop can use prange
        loop_analysis = self._get_loop_analysis(self._current_loop_index)
        self._current_loop_index += 1

        use_prange = False
        if loop_analysis is not None and loop_analysis.can_use_prange and self._in_jit_function:
            use_prange = True

        iterable = self._generate_expr(node.iterable)

        # Handle tuple destructuring in for loop: for (a, b) in ...
        if isinstance(node.variable, tuple):
            var_str = ", ".join(node.variable)
        else:
            var_str = node.variable

        if use_prange:
            # Convert range() to prange() for parallel execution
            # Handle both direct range expressions and range() calls
            prange_iterable = self._convert_to_prange(iterable, node.iterable)

            if self.verbose and loop_analysis is not None:
                # Add comment about parallelization
                if loop_analysis.reduction_vars:
                    reduction_list = ", ".join(sorted(loop_analysis.reduction_vars))
                    self._emit(f"# Parallel loop with reduction(s): {reduction_list}")
                else:
                    self._emit("# Parallel loop - iterations are independent")

            self._emit(f"for {var_str} in {prange_iterable}:")
        else:
            self._emit(f"for {var_str} in {iterable}:")

        self._indent()
        self.visit(node.body)
        self._dedent()

    def _convert_to_prange(self, iterable_str: str, iterable_node: Expression) -> str:
        """
        Convert a range expression or call to prange.

        Args:
            iterable_str: The generated string for the iterable
            iterable_node: The original AST node for the iterable

        Returns:
            The prange equivalent string
        """
        # Handle RangeExpression (0..10 or 0..=10 syntax)
        if isinstance(iterable_node, RangeExpression):
            start = self._generate_expr(iterable_node.start)
            end = self._generate_expr(iterable_node.end)
            if iterable_node.inclusive:
                return f"prange({start}, {end} + 1)"
            return f"prange({start}, {end})"

        # Handle CallExpression to range()
        if isinstance(iterable_node, CallExpression):
            if isinstance(iterable_node.callee, Identifier):
                if iterable_node.callee.name == "range":
                    # Replace range with prange, preserving arguments
                    args = ", ".join(
                        self._generate_expr(arg) for arg in iterable_node.arguments
                    )
                    return f"prange({args})"

        # Fallback: just replace "range(" with "prange(" in the string
        # This handles simple cases but may not work for complex expressions
        if iterable_str.startswith("range("):
            return "p" + iterable_str

        # If we can't convert, return original (shouldn't happen if analysis is correct)
        return iterable_str

    def visit_while_statement(self, node: WhileStatement) -> None:
        """Generate code for a while loop."""
        condition = self._generate_expr(node.condition)
        self._emit(f"while {condition}:")
        self._indent()
        self.visit(node.body)
        self._dedent()

    def visit_loop_statement(self, node: LoopStatement) -> None:
        """Generate code for an infinite loop."""
        self._emit("while True:")
        self._indent()
        self.visit(node.body)
        self._dedent()

    def visit_if_let_statement(self, node: "IfLetStatement") -> None:
        """
        Generate code for an if let statement.

        Example:
            if let Option::Some(x) = opt { body }
            ->
            _if_let_val = opt
            if isinstance(_if_let_val, Option_Some):
                x = _if_let_val._0
                body
        """
        from mathviz.compiler.ast_nodes import IfLetStatement

        # Generate a temporary variable for the value
        temp_var = "_if_let_val"
        value_code = self._generate_expr(node.value)
        self._emit(f"{temp_var} = {value_code}")

        # Generate the condition
        condition = self._generate_pattern_condition(node.pattern, temp_var)
        self._emit(f"if {condition}:")
        self._indent()

        # Generate bindings
        bindings = self._generate_pattern_bindings(node.pattern, temp_var)
        for name, expr in bindings.items():
            self._emit(f"{name} = {expr}")

        # Generate the then block
        self.visit(node.then_block)
        self._dedent()

        # Generate the else block if present
        if node.else_block:
            self._emit("else:")
            self._indent()
            self.visit(node.else_block)
            self._dedent()

    def visit_while_let_statement(self, node: "WhileLetStatement") -> None:
        """
        Generate code for a while let statement.

        Example:
            while let Option::Some(x) = iter.next() { body }
            ->
            while True:
                _while_let_val = iter.next()
                if not isinstance(_while_let_val, Option_Some):
                    break
                x = _while_let_val._0
                body
        """
        from mathviz.compiler.ast_nodes import WhileLetStatement

        temp_var = "_while_let_val"

        self._emit("while True:")
        self._indent()

        # Generate the value
        value_code = self._generate_expr(node.value)
        self._emit(f"{temp_var} = {value_code}")

        # Generate the break condition (negated pattern match)
        condition = self._generate_pattern_condition(node.pattern, temp_var)
        self._emit(f"if not ({condition}):")
        self._indent()
        self._emit("break")
        self._dedent()

        # Generate bindings
        bindings = self._generate_pattern_bindings(node.pattern, temp_var)
        for name, expr in bindings.items():
            self._emit(f"{name} = {expr}")

        # Generate the body
        self.visit(node.body)
        self._dedent()

    def visit_return_statement(self, node: ReturnStatement) -> None:
        """Generate code for a return statement."""
        if node.value:
            value = self._generate_expr(node.value)
            self._emit(f"return {value}")
        else:
            self._emit("return")

    def visit_break_statement(self, node: BreakStatement) -> None:
        """Generate code for a break statement."""
        self._emit("break")

    def visit_continue_statement(self, node: ContinueStatement) -> None:
        """Generate code for a continue statement."""
        self._emit("continue")

    def visit_pass_statement(self, node: PassStatement) -> None:
        """Generate code for a pass statement."""
        self._emit("pass")

    def visit_import_statement(self, node: ImportStatement) -> None:
        """Generate code for an import statement."""
        if node.is_from_import:
            names = ", ".join(
                f"{name} as {alias}" if alias else name
                for name, alias in node.names
            )
            self._emit(f"from {node.module} import {names}")
        else:
            if node.alias:
                self._emit(f"import {node.module} as {node.alias}")
            else:
                self._emit(f"import {node.module}")

    def visit_print_statement(self, node: PrintStatement) -> None:
        """Generate code for a print/println statement."""
        format_str = self._generate_expr(node.format_string)

        if node.arguments:
            args = ", ".join(self._generate_expr(arg) for arg in node.arguments)
            # Use f-string style formatting
            if node.newline:
                self._emit(f"print({format_str}.format({args}))")
            else:
                self._emit(f"print({format_str}.format({args}), end='')")
        else:
            if node.newline:
                self._emit(f"print({format_str})")
            else:
                self._emit(f"print({format_str}, end='')")

    def visit_use_statement(self, node: UseStatement) -> None:
        """Generate code for a use statement."""
        module_path = ".".join(node.module_path)

        # Skip if manim import is already handled
        if module_path == "manim" and self._needs_manim_import:
            return  # Already imported in header

        # Check if this is a MathViz module (loaded in registry)
        if self._module_registry and self._module_registry.is_loaded(module_path):
            # MathViz module - generate the module's code inline
            module_info = self._module_registry.get(module_path)
            if module_info and module_path not in self._generated_modules:
                self._generated_modules.add(module_path)
                self._emit(f"# MathViz module: {module_path}")
                self._generate_mviz_module(module_info)
            return

        # Check if this is a known Python module
        first_component = node.module_path[0] if node.module_path else ""
        if first_component in PYTHON_MODULES:
            # Standard Python import
            if node.wildcard:
                self._emit(f"from {module_path} import *")
            elif node.alias:
                self._emit(f"import {module_path} as {node.alias}")
            else:
                self._emit(f"import {module_path}")
        else:
            # Unknown module - generate as Python import with comment
            if node.wildcard:
                self._emit(f"from {module_path} import *")
            elif node.alias:
                self._emit(f"import {module_path} as {node.alias}")
            else:
                self._emit(f"import {module_path}")

    def _generate_mviz_module(self, module_info: "ModuleInfo") -> None:
        """
        Generate Python code for a MathViz module.

        The module is generated as a Python class with static methods,
        preserving the namespace for qualified access (e.g., module.function()).

        Args:
            module_info: The module information from the registry
        """
        from mathviz.compiler.module_loader import ModuleInfo

        # Use the last part of the module name as the class name
        class_name = module_info.name.split(".")[-1]

        self._emit(f"class {class_name}:")
        self._indent()

        has_content = False

        # Generate the module's content
        for stmt in module_info.ast.statements:
            # Skip use statements (already processed)
            if isinstance(stmt, UseStatement):
                continue

            # Generate functions as static methods
            if isinstance(stmt, FunctionDef):
                has_content = True
                self._emit("@staticmethod")
                self.visit(stmt)
            # Generate other statements normally
            elif isinstance(stmt, ModuleDecl):
                has_content = True
                self.visit(stmt)
            else:
                # For other statements (constants, etc.), include them
                has_content = True
                self.visit(stmt)

        if not has_content:
            self._emit("pass")

        self._dedent()
        self._emit("")

    def visit_module_decl(self, node: ModuleDecl) -> None:
        """Generate code for a module declaration (as a class with static methods)."""
        self._emit(f"class {node.name}:")
        self._indent()
        if node.body:
            # Mark functions as static
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    self._emit("@staticmethod")
                self.visit(stmt)
        else:
            self._emit("pass")
        self._dedent()
        self._emit("")

    def visit_play_statement(self, node: PlayStatement) -> None:
        """Generate code for a Manim play statement."""
        # Check if animation is a tuple literal (multiple animations)
        if isinstance(node.animation, TupleLiteral):
            # Unpack tuple elements as separate arguments
            animations = ", ".join(
                self._generate_expr(elem) for elem in node.animation.elements
            )
        else:
            animations = self._generate_expr(node.animation)

        if node.run_time:
            run_time = self._generate_expr(node.run_time)
            self._emit(f"self.play({animations}, run_time={run_time})")
        else:
            self._emit(f"self.play({animations})")

    def visit_wait_statement(self, node: WaitStatement) -> None:
        """Generate code for a Manim wait statement."""
        if node.duration:
            duration = self._generate_expr(node.duration)
            self._emit(f"self.wait({duration})")
        else:
            self._emit("self.wait()")

    # -------------------------------------------------------------------------
    # OOP Constructs
    # -------------------------------------------------------------------------

    def visit_struct_def(self, node: StructDef) -> None:
        """
        Generate code for a struct definition as a Python dataclass.

        Example MathViz:
            struct Point {
                x: Float
                y: Float
            }

        Generated Python:
            @dataclass
            class Point:
                x: float
                y: float

        For generic structs:
            struct Box<T> {
                value: T
            }

        Generated Python:
            T = TypeVar('T')

            @dataclass
            class Box(Generic[T]):
                value: T
        """
        # Handle generic type parameters
        if node.type_params:
            self._needs_typing_import = True
            # Generate TypeVar declarations for new type parameters
            for type_param in node.type_params:
                if type_param.name not in self._type_vars_declared:
                    self._type_vars_declared.add(type_param.name)
                    if type_param.bounds:
                        bounds_str = ", ".join(type_param.bounds)
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}', bound={bounds_str})")
                    else:
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}')")
            self._emit("")

        self._emit("@dataclass")

        if node.type_params:
            # Generic struct with type parameters
            type_params_str = ", ".join(tp.name for tp in node.type_params)
            self._emit(f"class {node.name}(Generic[{type_params_str}]):")
        else:
            self._emit(f"class {node.name}:")

        self._indent()

        if not node.fields:
            self._emit("pass")
        else:
            for field in node.fields:
                type_hint = self._generate_type(field.type_annotation)
                self._emit(f"{field.name}: {type_hint}")

        self._dedent()
        self._emit("")

    def visit_impl_block(self, node: ImplBlock) -> None:
        """
        Generate code for an impl block.

        For trait implementations, we generate methods that will be added to the class.
        For inherent implementations, we add methods directly to the class.

        In Python, we generate these as methods within the class definition itself
        or as standalone functions that operate on the type.

        Since Python classes need all methods at definition time, impl blocks
        are collected and merged with struct definitions during code generation.

        For operator traits (Add, Sub, Mul, etc.), we generate Python magic methods
        (__add__, __sub__, __mul__, etc.) so Python operators work natively.
        """
        from mathviz.compiler.operators import OPERATOR_TRAITS

        # Generate helper comment
        if node.trait_name:
            self._emit(f"# impl {node.trait_name} for {node.target_type}")
        else:
            self._emit(f"# impl {node.target_type}")

        # Check if this is an operator trait implementation
        trait_info = OPERATOR_TRAITS.get(node.trait_name) if node.trait_name else None

        # Generate each method as a standalone function that can be attached
        for method in node.methods:
            # Use Python magic method name for operator traits
            method_name = method.name
            if trait_info and method.name == trait_info.method_name:
                method_name = trait_info.python_magic
            else:
                # Both instance and static methods get prefixed with type name
                # Static: Type.method(args) -> Type_method(args)
                # Instance: obj.method(args) -> Type_method(obj, args)
                method_name = f"{node.target_type}_{method.name}"

            self._generate_method(method, node.target_type, method_name_override=method_name)
            self._emit("")

    def _generate_method(self, method: Method, target_type: str, method_name_override: Optional[str] = None) -> None:
        """Generate code for a single method.

        Args:
            method: The method AST node
            target_type: The type this method belongs to
            method_name_override: Optional override for method name (e.g., for magic methods)
        """
        # Build parameter list
        params: list[str] = []

        # Add self parameter if this is an instance method
        if method.has_self:
            params.append("self")

        for param in method.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {self._generate_type(param.type_annotation)}"
            if param.default_value:
                param_str += f" = {self._generate_expr(param.default_value)}"
            params.append(param_str)

        params_str = ", ".join(params)

        # Return type annotation
        return_type = ""
        if method.return_type:
            return_type = f" -> {self._generate_type(method.return_type)}"

        # Use override name if provided (for magic methods)
        method_name = method_name_override or method.name

        self._emit(f"def {method_name}({params_str}){return_type}:")
        self._indent()
        if method.body:
            self.visit(method.body)
        else:
            self._emit("pass")
        self._dedent()

    def visit_trait_def(self, node: TraitDef) -> None:
        """
        Generate code for a trait definition as a Python ABC.

        Example MathViz:
            trait Shape {
                fn area(self) -> Float
                fn perimeter(self) -> Float
            }

        Generated Python:
            from abc import ABC, abstractmethod

            class Shape(ABC):
                @abstractmethod
                def area(self) -> float:
                    pass

                @abstractmethod
                def perimeter(self) -> float:
                    pass

        For generic traits:
            trait Container<T> {
                fn get(self) -> T
                fn set(self, value: T)
            }

        Generated Python:
            T = TypeVar('T')

            class Container(ABC, Generic[T]):
                @abstractmethod
                def get(self) -> T:
                    pass
        """
        # Handle generic type parameters for traits
        if node.type_params:
            self._needs_typing_import = True
            # Generate TypeVar declarations for new type parameters
            for type_param in node.type_params:
                if type_param.name not in self._type_vars_declared:
                    self._type_vars_declared.add(type_param.name)
                    if type_param.bounds:
                        bounds_str = ", ".join(type_param.bounds)
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}', bound={bounds_str})")
                    else:
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}')")
            self._emit("")

        if node.type_params:
            type_params_str = ", ".join(tp.name for tp in node.type_params)
            self._emit(f"class {node.name}(ABC, Generic[{type_params_str}]):")
        else:
            self._emit(f"class {node.name}(ABC):")
        self._indent()

        if not node.methods:
            self._emit("pass")
        else:
            for method in node.methods:
                self._generate_trait_method(method)
                self._emit("")

        self._dedent()

    def _generate_trait_method(self, method: TraitMethod) -> None:
        """Generate code for a trait method."""
        # Build parameter list
        params: list[str] = []

        if method.has_self:
            params.append("self")

        for param in method.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str += f": {self._generate_type(param.type_annotation)}"
            params.append(param_str)

        params_str = ", ".join(params)

        # Return type annotation
        return_type = ""
        if method.return_type:
            return_type = f" -> {self._generate_type(method.return_type)}"

        if method.has_default_impl and method.default_body:
            # Method with default implementation
            self._emit(f"def {method.name}({params_str}){return_type}:")
            self._indent()
            self.visit(method.default_body)
            self._dedent()
        else:
            # Abstract method
            self._emit("@abstractmethod")
            self._emit(f"def {method.name}({params_str}){return_type}:")
            self._indent()
            self._emit("pass")
            self._dedent()

    def visit_enum_def(self, node: EnumDef) -> None:
        """
        Generate code for an enum definition.

        For simple enums (no associated data), use Python Enum.
        For enums with associated data, use dataclass-based tagged unions.
        For generic enums (Option<T>, Result<T, E>), use Generic base class.

        Example MathViz (generic with data):
            enum Option<T> {
                Some(T)
                None
            }

        Generated Python:
            T = TypeVar('T')

            class Option(Generic[T]):
                pass

            @dataclass
            class Some(Option[T]):
                _0: T

            @dataclass
            class _None(Option):  # None is reserved
                pass
        """
        # Handle generic type parameters
        if node.type_params:
            self._needs_typing_import = True
            # Generate TypeVar declarations for new type parameters
            for type_param in node.type_params:
                if type_param.name not in self._type_vars_declared:
                    self._type_vars_declared.add(type_param.name)
                    if type_param.bounds:
                        bounds_str = ", ".join(type_param.bounds)
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}', bound={bounds_str})")
                    else:
                        self._emit(f"{type_param.name} = TypeVar('{type_param.name}')")
            self._emit("")

        # Check if any variant has associated data
        has_data = any(variant.fields for variant in node.variants)

        if has_data:
            # Use dataclass-based tagged union
            if node.type_params:
                type_params_str = ", ".join(tp.name for tp in node.type_params)
                self._emit(f"class {node.name}(Generic[{type_params_str}]):")
            else:
                self._emit(f"class {node.name}:")
            self._indent()
            self._emit(f'"""Base class for {node.name} enum variants."""')
            self._emit("pass")
            self._dedent()
            self._emit("")

            for variant in node.variants:
                # Handle reserved keywords like None
                variant_name = variant.name
                if variant_name == "None":
                    variant_name = "_None"
                elif variant_name == "True":
                    variant_name = "_True"
                elif variant_name == "False":
                    variant_name = "_False"

                self._emit("@dataclass")
                if node.type_params and variant.fields:
                    # Generic variant with data inherits with type params
                    type_params_str = ", ".join(tp.name for tp in node.type_params)
                    self._emit(f"class {variant_name}({node.name}[{type_params_str}]):")
                else:
                    self._emit(f"class {variant_name}({node.name}):")
                self._indent()
                if variant.fields:
                    for i, field_type in enumerate(variant.fields):
                        type_hint = self._generate_type(field_type)
                        self._emit(f"_{i}: {type_hint}")
                else:
                    self._emit("pass")
                self._dedent()
                self._emit("")
        else:
            # Simple enum without associated data (can't be generic)
            self._emit(f"class {node.name}(Enum):")
            self._indent()
            for variant in node.variants:
                self._emit(f"{variant.name} = auto()")
            self._dedent()
            self._emit("")

    def visit_self_expression(self, node: SelfExpression) -> str:
        """Generate code for a self expression."""
        return "self"

    def visit_enum_variant_access(self, node: EnumVariantAccess) -> str:
        """
        Generate code for enum variant access or struct static method.

        Example:
            Shape::Circle -> Circle (for dataclass variants with data)
            Color::Red -> Color.Red (for Python Enum without data)
            Point::new -> Point_new (for struct static methods)
            Option::None -> _None() (None is a reserved keyword)
        """
        # Handle reserved keywords - must match the transformation in visit_enum_def
        variant_name = node.variant_name
        if variant_name == "None":
            variant_name = "_None"
        elif variant_name == "True":
            variant_name = "_True"
        elif variant_name == "False":
            variant_name = "_False"

        # Check if this is an enum variant access
        if node.enum_name in self._enum_variants:
            if node.variant_name in self._enum_variants[node.enum_name]:
                # It's an enum variant
                if node.enum_name in self._simple_enums:
                    # Simple enum (Python Enum) - use EnumName.VariantName
                    return f"{node.enum_name}.{node.variant_name}"
                elif (node.enum_name, node.variant_name) in self._enum_variants_with_data:
                    # Variant has data - will be called as function, just return class name
                    return variant_name
                else:
                    # Dataclass-based enum variant without data - create instance
                    return f"{variant_name}()"

        # Otherwise it's a static method from an impl block
        return f"{node.enum_name}_{node.variant_name}"

    def visit_struct_literal(self, node: StructLiteral) -> str:
        """
        Generate code for a struct literal.

        Example:
            Point { x: 1.0, y: 2.0 } -> Point(x=1.0, y=2.0)
            Point { x: 10.0, ...p1 } -> replace(p1, x=10.0)
        """
        if node.spread:
            # Struct update syntax using dataclasses.replace
            self._needs_dataclass_replace = True
            spread_expr = self._generate_expr(node.spread)
            if node.fields:
                field_args = ", ".join(
                    f"{name}={self._generate_expr(value)}"
                    for name, value in node.fields
                )
                return f"replace({spread_expr}, {field_args})"
            else:
                # Just copy the struct
                return f"replace({spread_expr})"
        else:
            field_args = ", ".join(
                f"{name}={self._generate_expr(value)}"
                for name, value in node.fields
            )
            return f"{node.struct_name}({field_args})"

    def visit_enum_pattern(self, node: EnumPattern) -> str:
        """Generate condition for enum pattern matching."""
        # This is used internally by pattern matching code generation
        # The actual pattern matching generates isinstance checks
        return f"{node.enum_name}.{node.variant_name}"


def generate(
    program: Program,
    optimize: bool = True,
    purity_info: dict[str, PurityInfo] | None = None,
    complexity_info: dict[str, ComplexityInfo] | None = None,
    parallel_info: dict[str, list[LoopAnalysis]] | None = None,
    call_graph: CallGraph | None = None,
    verbose: bool = False,
) -> str:
    """
    Convenience function to generate Python code from an AST.

    Args:
        program: The root Program node
        optimize: Whether to inject Numba JIT decorators
        purity_info: Purity analysis results from PurityAnalyzer
        complexity_info: Complexity analysis results from ComplexityAnalyzer
        parallel_info: Parallelization analysis from ParallelAnalyzer
        call_graph: Call graph from CallGraphBuilder
        verbose: Whether to emit complexity comments

    Returns:
        Generated Python source code

    Example:
        # Basic usage (backward compatible)
        code = generate(ast)

        # With analysis data for smarter optimization
        from mathviz.compiler.purity_analyzer import analyze_purity
        from mathviz.compiler.parallel_analyzer import ParallelAnalyzer
        from mathviz.compiler.complexity_analyzer import analyze_complexity
        from mathviz.compiler.call_graph import CallGraphBuilder

        purity = analyze_purity(ast)
        complexity = analyze_complexity(ast)

        analyzer = ParallelAnalyzer()
        parallel = {}
        for stmt in ast.statements:
            if isinstance(stmt, FunctionDef):
                loops = analyzer.analyze_function(stmt)
                parallel[stmt.name] = [analysis for _, analysis in loops]

        builder = CallGraphBuilder()
        call_graph = builder.build(ast)

        code = generate(
            ast,
            purity_info=purity,
            complexity_info=complexity,
            parallel_info=parallel,
            call_graph=call_graph,
            verbose=True,
        )
    """
    generator = CodeGenerator(
        optimize=optimize,
        purity_info=purity_info,
        complexity_info=complexity_info,
        parallel_info=parallel_info,
        call_graph=call_graph,
        verbose=verbose,
    )
    return generator.generate(program)
