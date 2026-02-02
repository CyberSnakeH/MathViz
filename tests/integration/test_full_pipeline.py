"""
Integration tests for the complete MathViz compilation pipeline.

These tests verify that all analyzers work together correctly in the
full compilation pipeline, from parsing through code generation.
"""

import pytest

from mathviz.compiler.purity_analyzer import Purity, is_jit_safe
from mathviz.compiler.complexity_analyzer import Complexity


class TestCompilationPipeline:
    """Test the full compilation pipeline with all analyzers."""

    def test_pipeline_basic_program(self, compile_with_analysis):
        """Test basic program compiles with all analyses."""
        source = """
fn add(a: Int, b: Int) -> Int {
    return a + b
}

fn multiply(x: Int, y: Int) -> Int {
    return x * y
}

let result = add(5, 3)
"""
        result = compile_with_analysis(source)

        # Verify AST was generated
        assert result.ast is not None
        assert len(result.ast.statements) > 0

        # Verify both code versions were generated
        assert result.python_code is not None
        assert result.optimized_code is not None
        assert "def add" in result.python_code
        assert "def multiply" in result.python_code

        # Verify no type errors for valid program
        assert not result.has_type_errors(), f"Unexpected errors: {result.type_errors}"

        # Verify purity analysis ran
        assert "add" in result.purity_info
        assert "multiply" in result.purity_info
        assert result.purity_info["add"].is_pure()
        assert result.purity_info["multiply"].is_pure()

        # Verify complexity analysis ran
        assert "add" in result.complexity_info
        assert result.complexity_info["add"].complexity == Complexity.O_1

        # Verify call graph was built
        assert result.call_graph is not None
        assert "add" in result.call_graph.nodes
        assert "multiply" in result.call_graph.nodes

    def test_pipeline_type_errors_detected(self, compile_with_analysis):
        """Test that type errors are caught during compilation."""
        source = """
fn add_numbers(a: Int, b: Int) -> Int {
    return a + b
}

let x: Int = "hello"
"""
        result = compile_with_analysis(source)

        # Should have type errors
        assert result.has_type_errors()

        # Verify the error is about type mismatch
        error_messages = [str(e) for e in result.type_errors]
        assert any("Int" in msg or "String" in msg for msg in error_messages)

        # Despite errors, other analyses should still complete
        assert "add_numbers" in result.purity_info
        assert result.call_graph is not None

    def test_pipeline_purity_affects_jit(self, compile_with_analysis):
        """Test that impure functions don't get JIT decorators."""
        source = """
fn pure_math(x: Int, y: Int) -> Int {
    return x * y + x
}

fn impure_with_print(x: Int) -> Int {
    println("Processing: {}", x)
    return x * 2
}

fn impure_with_global() -> Int {
    counter = counter + 1
    return counter
}
"""
        result = compile_with_analysis(source, optimize=True)

        # Pure function should be JIT-safe
        assert "pure_math" in result.purity_info
        pure_info = result.purity_info["pure_math"]
        assert pure_info.is_pure()
        assert is_jit_safe(pure_info)

        # Print function should be impure (I/O)
        assert "impure_with_print" in result.purity_info
        print_info = result.purity_info["impure_with_print"]
        assert print_info.purity == Purity.IMPURE_IO
        assert print_info.has_io()

        # Global mutation should be impure
        assert "impure_with_global" in result.purity_info
        global_info = result.purity_info["impure_with_global"]
        assert global_info.purity == Purity.IMPURE_MUTATION
        assert global_info.has_mutations()

        # Verify optimized code has JIT only for pure function
        # Pure function should get JIT decorator in optimized code
        assert "pure_math" in result.get_jit_candidates()

    def test_pipeline_auto_parallelization(self, compile_with_analysis):
        """Test that parallelizable loops are detected."""
        source = """
fn sum_array(arr: List[Int], n: Int) -> Int {
    let total = 0
    for i in range(n) {
        total += arr[i]
    }
    return total
}

fn vector_add(a: List[Int], b: List[Int], n: Int) -> List[Int] {
    let result = zeros(n)
    for i in range(n) {
        result[i] = a[i] + b[i]
    }
    return result
}

fn dependent_loop(arr: List[Int], n: Int) -> Int {
    for i in range(1, n) {
        arr[i] = arr[i] + arr[i-1]
    }
    return arr[n-1]
}
"""
        result = compile_with_analysis(source)

        # Check loop analyses were performed
        assert len(result.parallel_loops) >= 3

        # Find the analyses by function name
        analyses_by_func = {}
        for func_name, analysis in result.parallel_loops:
            if func_name not in analyses_by_func:
                analyses_by_func[func_name] = []
            analyses_by_func[func_name].append(analysis)

        # sum_array has a reduction pattern (total +=)
        assert "sum_array" in analyses_by_func
        sum_analysis = analyses_by_func["sum_array"][0]
        assert "total" in sum_analysis.reduction_vars or sum_analysis.is_parallelizable

        # vector_add should be parallelizable (independent iterations)
        assert "vector_add" in analyses_by_func
        vector_analysis = analyses_by_func["vector_add"][0]
        assert vector_analysis.is_parallelizable

        # dependent_loop should NOT be parallelizable (arr[i-1] dependency)
        assert "dependent_loop" in analyses_by_func
        dep_analysis = analyses_by_func["dependent_loop"][0]
        assert not dep_analysis.is_parallelizable or len(dep_analysis.dependencies) > 0

    def test_pipeline_call_graph_ordering(self, compile_with_analysis):
        """Test functions are ordered by call graph dependencies."""
        source = """
fn helper() -> Int {
    return 42
}

fn middle() -> Int {
    return helper() + 10
}

fn main() -> Int {
    return middle() * 2
}
"""
        result = compile_with_analysis(source)

        # Verify call graph structure
        assert result.call_graph is not None
        graph = result.call_graph

        # Check call relationships
        assert "helper" in graph.nodes
        assert "middle" in graph.nodes
        assert "main" in graph.nodes

        # middle calls helper
        assert "helper" in graph.get_callees("middle")

        # main calls middle
        assert "middle" in graph.get_callees("main")

        # Get topological order (helper should come before middle, middle before main)
        order = result.get_compilation_order()

        # All functions should be in the order
        assert "helper" in order
        assert "middle" in order
        assert "main" in order

        # helper should appear before middle (which calls it)
        assert order.index("helper") < order.index("middle")

        # middle should appear before main (which calls it)
        assert order.index("middle") < order.index("main")

    def test_pipeline_recursion_detection(self, compile_with_analysis):
        """Test that recursive functions are properly identified."""
        source = """
fn factorial(n: Int) -> Int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

fn fib(n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

fn not_recursive(x: Int) -> Int {
    return x * 2
}
"""
        result = compile_with_analysis(source)

        # Get recursive functions
        recursive = result.get_recursive_functions()

        # factorial and fib should be identified as recursive
        assert "factorial" in recursive
        assert "fib" in recursive

        # not_recursive should not be in the list
        assert "not_recursive" not in recursive

        # Verify complexity analysis detects recursion
        assert "factorial" in result.complexity_info
        factorial_complexity = result.complexity_info["factorial"]
        assert factorial_complexity.has_recursion
        assert factorial_complexity.recursive_calls >= 1

        assert "fib" in result.complexity_info
        fib_complexity = result.complexity_info["fib"]
        assert fib_complexity.has_recursion
        # Fib has 2 recursive calls
        assert fib_complexity.recursive_calls == 2
        # Fib should be detected as exponential
        assert fib_complexity.complexity == Complexity.O_2_N

    def test_pipeline_complexity_nested_loops(self, compile_with_analysis):
        """Test complexity detection for nested loops."""
        source = """
fn constant_time(x: Int) -> Int {
    return x * 2 + 1
}

fn linear(arr: List[Int], n: Int) -> Int {
    let sum = 0
    for i in range(n) {
        sum += arr[i]
    }
    return sum
}

fn quadratic(mat: List[List[Int]], n: Int) -> Int {
    let sum = 0
    for i in range(n) {
        for j in range(n) {
            sum += mat[i][j]
        }
    }
    return sum
}

fn cubic(tensor: List[List[List[Int]]], n: Int) -> Int {
    let sum = 0
    for i in range(n) {
        for j in range(n) {
            for k in range(n) {
                sum += tensor[i][j][k]
            }
        }
    }
    return sum
}
"""
        result = compile_with_analysis(source)

        # Verify complexity classifications
        assert result.complexity_info["constant_time"].complexity == Complexity.O_1
        assert result.complexity_info["linear"].complexity == Complexity.O_N
        assert result.complexity_info["quadratic"].complexity == Complexity.O_N_SQUARED
        assert result.complexity_info["cubic"].complexity == Complexity.O_N_CUBED

        # Verify loop depths
        assert result.complexity_info["constant_time"].loop_depth == 0
        assert result.complexity_info["linear"].loop_depth == 1
        assert result.complexity_info["quadratic"].loop_depth == 2
        assert result.complexity_info["cubic"].loop_depth == 3

    def test_pipeline_scene_not_jit(self, compile_with_analysis):
        """Test that Manim scenes are identified as impure and not JIT'd."""
        source = """
fn pure_helper(x: Float) -> Float {
    return x * 2.0
}

scene MyAnimation {
    fn construct(self) {
        let circle = Circle()
        play(Create(circle))
        wait(1.0)
    }
}
"""
        result = compile_with_analysis(source)

        # Pure helper should be pure
        assert "pure_helper" in result.purity_info
        assert result.purity_info["pure_helper"].is_pure()

        # Scene should be identified as Manim/impure
        assert "MyAnimation" in result.purity_info
        scene_info = result.purity_info["MyAnimation"]
        assert scene_info.purity == Purity.IMPURE_MANIM
        assert scene_info.has_manim_calls()
        assert not is_jit_safe(scene_info)

        # Generated code should have Manim imports
        assert "from manim import *" in result.python_code

    def test_pipeline_all_analyses_complete(self, compile_with_analysis):
        """Test that all analysis phases complete without errors."""
        source = """
fn helper(x: Int) -> Int {
    return x + 1
}

fn process(arr: List[Int], n: Int) -> Int {
    let result = 0
    for i in range(n) {
        result += helper(arr[i])
    }
    return result
}

class Calculator {
    fn __init__(self, value: Int) {
        self.value = value
    }

    fn add(self, x: Int) -> Int {
        return self.value + x
    }
}
"""
        result = compile_with_analysis(source)

        # All analyses should have run successfully
        assert result.ast is not None
        assert result.python_code is not None
        assert result.optimized_code is not None
        assert result.type_checker is not None
        assert result.call_graph is not None

        # No unexpected type errors
        assert not result.has_type_errors()

        # Purity analysis completed
        assert "helper" in result.purity_info
        assert "process" in result.purity_info

        # Complexity analysis completed
        assert "helper" in result.complexity_info
        assert "process" in result.complexity_info

        # Call graph populated
        assert "helper" in result.call_graph.nodes
        assert "process" in result.call_graph.nodes

        # Parallelization analysis ran
        assert len(result.parallel_loops) >= 1


class TestPipelineCodeGeneration:
    """Test that analysis results correctly influence code generation."""

    def test_jit_decorator_applied_to_pure_functions(self, compile_with_analysis):
        """Test that JIT decorators are added to pure numeric functions."""
        source = """
fn compute(x: Int, y: Int) -> Int {
    return x * x + y * y
}
"""
        result = compile_with_analysis(source, optimize=True)

        # Pure numeric function should get JIT in optimized code
        assert "@njit" in result.optimized_code or "@jit" in result.optimized_code

        # Non-optimized code should not have JIT
        assert "@njit" not in result.python_code
        assert "@jit" not in result.python_code

    def test_numpy_imports_added_when_needed(self, compile_with_analysis):
        """Test that numpy imports are added for JIT functions."""
        source = """
fn vector_dot(a: List[Float], b: List[Float], n: Int) -> Float {
    let result = 0.0
    for i in range(n) {
        result += a[i] * b[i]
    }
    return result
}
"""
        result = compile_with_analysis(source, optimize=True)

        # Optimized code should have numpy import for JIT compatibility
        assert "import numpy" in result.optimized_code or "from numba" in result.optimized_code

    def test_manim_imports_for_scenes(self, compile_with_analysis):
        """Test that Manim imports are added for scene definitions."""
        source = """
scene TestScene {
    fn construct(self) {
        let square = Square()
        play(Create(square))
    }
}
"""
        result = compile_with_analysis(source)

        # Should have manim import
        assert "from manim import *" in result.python_code
        assert "from manim import *" in result.optimized_code

        # Scene should inherit from Scene
        assert "class TestScene(Scene):" in result.python_code

    def test_runtime_imports_for_set_operations(self, compile_with_analysis):
        """Test that runtime imports are added for mathematical set operations."""
        source = """
let A = {1, 2, 3}
let B = {2, 3, 4}
let union_result = A \u222a B
let intersection_result = A \u2229 B
"""
        result = compile_with_analysis(source)

        # Should have runtime imports for set operations
        assert "set_union" in result.python_code
        assert "set_intersection" in result.python_code


class TestPipelineEdgeCases:
    """Test edge cases and error handling in the pipeline."""

    def test_empty_program(self, compile_with_analysis):
        """Test compilation of an empty program."""
        source = ""
        result = compile_with_analysis(source)

        assert result.ast is not None
        assert len(result.type_errors) == 0
        assert len(result.purity_info) == 0
        assert len(result.complexity_info) == 0

    def test_function_with_no_body(self, compile_with_analysis):
        """Test function with pass statement."""
        source = """
fn empty_func() {
    pass
}
"""
        result = compile_with_analysis(source)

        assert "empty_func" in result.purity_info
        assert result.purity_info["empty_func"].is_pure()
        assert result.complexity_info["empty_func"].complexity == Complexity.O_1

    def test_mutually_recursive_functions(self, compile_with_analysis):
        """Test detection of mutual recursion."""
        source = """
fn is_even(n: Int) -> Bool {
    if n == 0 {
        return true
    }
    return is_odd(n - 1)
}

fn is_odd(n: Int) -> Bool {
    if n == 0 {
        return false
    }
    return is_even(n - 1)
}
"""
        result = compile_with_analysis(source)

        # Both functions should be in call graph
        assert "is_even" in result.call_graph.nodes
        assert "is_odd" in result.call_graph.nodes

        # They should call each other
        assert "is_odd" in result.call_graph.get_callees("is_even")
        assert "is_even" in result.call_graph.get_callees("is_odd")

        # They should be detected as being in a cycle
        assert result.call_graph.nodes["is_even"].in_cycle
        assert result.call_graph.nodes["is_odd"].in_cycle

        # Topological sort should fail or handle cycle
        cycles = result.call_graph.find_cycles()
        assert len(cycles) > 0

    def test_deeply_nested_structures(self, compile_with_analysis):
        """Test compilation of deeply nested control structures."""
        source = """
fn deeply_nested(x: Int) -> Int {
    if x > 0 {
        if x > 10 {
            if x > 100 {
                for i in range(x) {
                    while i > 0 {
                        if i % 2 == 0 {
                            return i
                        }
                        i = i - 1
                    }
                }
            }
        }
    }
    return 0
}
"""
        result = compile_with_analysis(source)

        # Should compile without errors
        assert not result.has_type_errors()
        assert "deeply_nested" in result.purity_info
        assert "deeply_nested" in result.complexity_info

        # Should detect the nested loop structure
        assert result.complexity_info["deeply_nested"].loop_depth >= 1
