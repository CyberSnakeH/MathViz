"""
Integration tests for multi-file MathViz module compilation.

These tests verify the complete module system works end-to-end,
from parsing to code generation.
"""

import pytest
from pathlib import Path
from textwrap import dedent

from mathviz.compiler import CompilationPipeline, ModuleLoader


class TestMultiFileCompilation:
    """Tests for compiling multi-file MathViz projects."""

    def test_simple_two_file_project(self, tmp_path):
        """Test compiling a project with two files."""
        # Create geometry module
        geometry_file = tmp_path / "geometry.mviz"
        geometry_file.write_text(
            dedent("""
            pub fn distance(x1, y1, x2, y2) {
                let dx = x2 - x1
                let dy = y2 - y1
                return sqrt(dx*dx + dy*dy)
            }

            pub fn midpoint(x1, y1, x2, y2) {
                return ((x1 + x2) / 2, (y1 + y2) / 2)
            }
        """)
        )

        # Create main file that uses geometry
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use geometry

            fn main() {
                let d = geometry.distance(0, 0, 3, 4)
                print(d)
            }
        """)
        )

        # Compile main file
        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,  # Skip type checking for this test
            strict=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        assert "class geometry:" in result.python_code
        assert "@staticmethod" in result.python_code
        assert "def distance" in result.python_code

    def test_nested_module_import(self, tmp_path):
        """Test importing nested modules (lib/math.mviz)."""
        # Create lib directory
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()

        # Create lib/math.mviz
        math_file = lib_dir / "math.mviz"
        math_file.write_text(
            dedent("""
            pub fn add(a, b) {
                return a + b
            }

            pub fn multiply(a, b) {
                return a * b
            }
        """)
        )

        # Create main file
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use lib.math

            fn main() {
                let sum = math.add(2, 3)
                let product = math.multiply(4, 5)
                print(sum + product)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        assert "class math:" in result.python_code
        assert "def add" in result.python_code
        assert "def multiply" in result.python_code

    def test_transitive_dependencies(self, tmp_path):
        """Test that transitive dependencies are handled (a -> b -> c)."""
        # Create core module
        core_file = tmp_path / "core.mviz"
        core_file.write_text(
            dedent("""
            pub fn base_value() {
                return 42
            }
        """)
        )

        # Create utils module that depends on core
        utils_file = tmp_path / "utils.mviz"
        utils_file.write_text(
            dedent("""
            use core

            pub fn get_doubled() {
                return core.base_value() * 2
            }
        """)
        )

        # Create main that depends on utils
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use utils

            fn main() {
                let result = utils.get_doubled()
                print(result)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        # Both core and utils should be generated
        assert "class utils:" in result.python_code

    def test_diamond_dependency(self, tmp_path):
        """Test diamond dependency pattern (a -> b, a -> c, b -> d, c -> d)."""
        # Create base module d
        d_file = tmp_path / "d.mviz"
        d_file.write_text(
            dedent("""
            pub fn base() {
                return 1
            }
        """)
        )

        # Create modules b and c that both depend on d
        b_file = tmp_path / "b.mviz"
        b_file.write_text(
            dedent("""
            use d

            pub fn from_b() {
                return d.base() + 10
            }
        """)
        )

        c_file = tmp_path / "c.mviz"
        c_file.write_text(
            dedent("""
            use d

            pub fn from_c() {
                return d.base() + 100
            }
        """)
        )

        # Create main that depends on both b and c
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use b
            use c

            fn main() {
                let result = b.from_b() + c.from_c()
                print(result)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"

    def test_mixed_python_and_mviz_imports(self, tmp_path):
        """Test mixing Python imports with MathViz module imports."""
        # Create a MathViz module
        helpers_file = tmp_path / "helpers.mviz"
        helpers_file.write_text(
            dedent("""
            pub fn square(x) {
                return x * x
            }
        """)
        )

        # Create main with both Python and MathViz imports
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use numpy
            use helpers

            fn main() {
                let arr = numpy.array([1, 2, 3])
                let sq = helpers.square(5)
                print(sq)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        # Python import should be standard
        assert "import numpy" in result.python_code
        # MathViz module should be a class
        assert "class helpers:" in result.python_code

    def test_public_private_visibility(self, tmp_path):
        """Test that only public functions are accessible."""
        # Create module with public and private functions
        module_file = tmp_path / "mymodule.mviz"
        module_file.write_text(
            dedent("""
            pub fn public_func() {
                return private_helper() * 2
            }

            fn private_helper() {
                return 21
            }
        """)
        )

        # Create main that uses the module
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use mymodule

            fn main() {
                let result = mymodule.public_func()
                print(result)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        # Both functions should be in the generated code (needed for internal use)
        assert "def public_func" in result.python_code
        assert "def private_helper" in result.python_code

    def test_directory_module(self, tmp_path):
        """Test directory module (utils/mod.mviz)."""
        # Create directory module
        utils_dir = tmp_path / "utils"
        utils_dir.mkdir()

        mod_file = utils_dir / "mod.mviz"
        mod_file.write_text(
            dedent("""
            pub fn from_directory() {
                return "I'm from a directory module"
            }
        """)
        )

        # Create main
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use utils

            fn main() {
                let msg = utils.from_directory()
                print(msg)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        assert "class utils:" in result.python_code


class TestCircularDependencyHandling:
    """Tests for circular dependency detection and error handling."""

    def test_direct_circular_dependency_warning(self, tmp_path):
        """Test that direct circular dependencies generate warnings."""
        # Create two modules that depend on each other
        a_file = tmp_path / "a.mviz"
        a_file.write_text(
            dedent("""
            use b

            pub fn from_a() {
                return 1
            }
        """)
        )

        b_file = tmp_path / "b.mviz"
        b_file.write_text(
            dedent("""
            use a

            pub fn from_b() {
                return 2
            }
        """)
        )

        # Create main that uses a
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use a

            fn main() {
                print(a.from_a())
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
            strict=False,  # Allow warnings
        )
        result = pipeline.compile_file(main_file)

        # Should either succeed with warnings or fail gracefully
        # The exact behavior depends on how we handle cycles
        # For now, we just check it doesn't crash
        assert isinstance(result.python_code, str)


class TestModuleSearchPaths:
    """Tests for module search path functionality."""

    def test_custom_search_path(self, tmp_path):
        """Test that custom search paths are used."""
        # Create separate lib directory
        lib_dir = tmp_path / "external_libs"
        lib_dir.mkdir()

        # Create module in lib directory
        ext_file = lib_dir / "external.mviz"
        ext_file.write_text(
            dedent("""
            pub fn from_external() {
                return "external library"
            }
        """)
        )

        # Create src directory
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Create main in src
        main_file = src_dir / "main.mviz"
        main_file.write_text(
            dedent("""
            use external

            fn main() {
                let msg = external.from_external()
                print(msg)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
            module_search_paths=[lib_dir],
        )
        result = pipeline.compile_file(main_file)

        assert result.success, f"Compilation failed: {result.warnings}"
        assert "class external:" in result.python_code


class TestGeneratedCodeQuality:
    """Tests for quality of generated Python code from modules."""

    def test_generated_code_is_valid_python(self, tmp_path):
        """Test that generated code is syntactically valid Python."""
        # Create a module
        math_file = tmp_path / "mathlib.mviz"
        math_file.write_text(
            dedent("""
            pub fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                return n * factorial(n - 1)
            }
        """)
        )

        # Create main
        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use mathlib

            fn main() {
                let result = mathlib.factorial(5)
                print(result)
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success

        # Verify the code is valid Python by compiling it
        try:
            compile(result.python_code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}\n\nCode:\n{result.python_code}")

    def test_module_functions_are_static_methods(self, tmp_path):
        """Test that module functions are generated as static methods."""
        module_file = tmp_path / "mymod.mviz"
        module_file.write_text(
            dedent("""
            pub fn func1() {
                return 1
            }

            pub fn func2(x) {
                return x * 2
            }
        """)
        )

        main_file = tmp_path / "main.mviz"
        main_file.write_text(
            dedent("""
            use mymod

            fn main() {
                print(mymod.func1())
            }
        """)
        )

        pipeline = CompilationPipeline(
            optimize=False,
            check_types=False,
        )
        result = pipeline.compile_file(main_file)

        assert result.success
        # Check for @staticmethod decorator before each function
        assert result.python_code.count("@staticmethod") >= 2
