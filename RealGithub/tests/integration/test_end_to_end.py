"""
End-to-end integration tests for MathViz compiler.

These tests verify that complete MathViz programs compile correctly
and produce executable Python code.
"""

import pytest
import sys
from io import StringIO

from mathviz import compile_source


def execute_code(python_code: str, globals_dict: dict | None = None) -> dict:
    """
    Execute compiled Python code and return the resulting namespace.

    This is a helper for testing - it executes the generated code in a
    controlled environment.
    """
    exec_globals = globals_dict or {}
    # Using exec() here is intentional for testing purposes only.
    # This executes compiler-generated code, not user input.
    compiled = compile(python_code, "<test>", "exec")
    eval(compiled, exec_globals)  # noqa: S307 - safe, compiler output only
    return exec_globals


class TestEndToEndCompilation:
    """End-to-end compilation tests."""

    def test_simple_program_executes(self):
        """Test that a simple compiled program executes correctly."""
        source = """
let x = 10
let y = 20
let z = x + y
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["x"] == 10
        assert result["y"] == 20
        assert result["z"] == 30

    def test_function_definition_and_call(self):
        """Test function compilation and execution."""
        source = """
fn add(a: Int, b: Int) -> Int {
    return a + b
}

let result = add(5, 3)
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["result"] == 8
        assert callable(result["add"])
        assert result["add"](10, 20) == 30

    def test_control_flow(self):
        """Test control flow statements."""
        source = """
fn max_of_three(a: Int, b: Int, c: Int) -> Int {
    let result = a
    if b > result {
        result = b
    }
    if c > result {
        result = c
    }
    return result
}

let maximum = max_of_three(5, 9, 3)
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["maximum"] == 9

    def test_for_loop(self):
        """Test for loop execution."""
        source = """
let total = 0
for i in [1, 2, 3, 4, 5] {
    total = total + i
}
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["total"] == 15

    def test_while_loop(self):
        """Test while loop execution."""
        source = """
let count = 0
let sum = 0
while count < 5 {
    count = count + 1
    sum = sum + count
}
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["count"] == 5
        assert result["sum"] == 15

    def test_nested_control_flow(self):
        """Test nested control flow."""
        source = """
let result = 0
for i in [1, 2, 3] {
    for j in [1, 2, 3] {
        if i == j {
            result = result + 1
        }
    }
}
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["result"] == 3  # i==j happens 3 times

    def test_class_definition(self):
        """Test class compilation."""
        source = """
class Point {
    fn __init__(self, x: Float, y: Float) {
        self.x = x
        self.y = y
    }

    fn distance_from_origin(self) -> Float {
        return (self.x ^ 2 + self.y ^ 2) ^ 0.5
    }
}

let p = Point(3.0, 4.0)
let dist = p.distance_from_origin()
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert abs(result["dist"] - 5.0) < 0.0001


class TestMathOperationsExecution:
    """Test mathematical operations compile and execute correctly."""

    def test_set_operations_with_runtime(self):
        """Test that set operations use runtime and execute correctly."""
        source = """
let A = {1, 2, 3}
let B = {2, 3, 4}

let union_AB = A ∪ B
let intersection_AB = A ∩ B
let diff_AB = A ∖ B
"""
        python_code = compile_source(source, optimize=False)

        # Need to make runtime available
        from mathviz.runtime import math_ops

        exec_globals = {
            "set_union": math_ops.set_union,
            "set_intersection": math_ops.set_intersection,
            "set_difference": math_ops.set_difference,
            "is_element": math_ops.is_element,
            "is_not_element": math_ops.is_not_element,
            "is_subset": math_ops.is_subset,
            "is_superset": math_ops.is_superset,
        }
        result = execute_code(python_code, exec_globals)

        assert result["union_AB"] == {1, 2, 3, 4}
        assert result["intersection_AB"] == {2, 3}
        assert result["diff_AB"] == {1}

    def test_membership_operations(self):
        """Test membership operations execute correctly."""
        source = """
let S = {1, 2, 3, 4, 5}
let member_check = 3 ∈ S
let non_member_check = 10 ∉ S
"""
        python_code = compile_source(source, optimize=False)

        from mathviz.runtime import math_ops

        exec_globals = {
            "is_element": math_ops.is_element,
            "is_not_element": math_ops.is_not_element,
        }
        result = execute_code(python_code, exec_globals)

        assert result["member_check"] is True
        assert result["non_member_check"] is True

    def test_subset_operations(self):
        """Test subset/superset operations execute correctly."""
        source = """
let A = {1, 2}
let B = {1, 2, 3, 4}
let is_A_subset_B = A ⊆ B
let is_B_superset_A = B ⊇ A
"""
        python_code = compile_source(source, optimize=False)

        from mathviz.runtime import math_ops

        exec_globals = {
            "is_subset": math_ops.is_subset,
            "is_superset": math_ops.is_superset,
        }
        result = execute_code(python_code, exec_globals)

        assert result["is_A_subset_B"] is True
        assert result["is_B_superset_A"] is True

    def test_exponentiation(self):
        """Test exponentiation with both ^ and **."""
        source = """
let a = 2 ^ 10
let b = 3 ** 4
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["a"] == 1024
        assert result["b"] == 81


class TestListAndDictOperations:
    """Test list and dictionary operations."""

    def test_list_operations(self):
        """Test list literal and indexing."""
        source = """
let items = [10, 20, 30, 40]
let first = items[0]
let last = items[3]
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["first"] == 10
        assert result["last"] == 40

    def test_dict_operations(self):
        """Test dictionary literal and access."""
        source = """
let data = {"name": "Alice", "age": 30}
let name = data["name"]
"""
        python_code = compile_source(source, optimize=False)
        result = execute_code(python_code)

        assert result["name"] == "Alice"


class TestErrorHandlingIntegration:
    """Test error handling in compiled code."""

    def test_runtime_error_propagates(self):
        """Test that runtime errors propagate correctly."""
        source = """
let items = [1, 2, 3]
let bad = items[10]
"""
        python_code = compile_source(source, optimize=False)

        with pytest.raises(IndexError):
            execute_code(python_code)
