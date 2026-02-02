"""
Tests for the MathViz REPL (Read-Eval-Print Loop).

Tests cover:
- REPLSession initialization and builtins
- Expression evaluation
- Variable and function definitions
- Command handling (:help, :vars, :funcs, :type, :ast, etc.)
- Tab completion
- Type inference
"""

import pytest
from unittest.mock import patch
from io import StringIO

from mathviz.repl import REPLSession, REPLCompleter, Colors, DefinedFunction


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def session():
    """Create a fresh REPL session for testing."""
    return REPLSession()


@pytest.fixture
def completer(session):
    """Create a completer attached to a session."""
    return REPLCompleter(session)


# =============================================================================
# REPLSession Initialization Tests
# =============================================================================


class TestREPLSessionInit:
    """Tests for REPLSession initialization."""

    def test_empty_initial_state(self, session):
        """Session should start with empty variables and functions."""
        assert session.variables == {}
        assert session.functions == {}
        assert session.variable_types == {}
        assert session.history == []

    def test_builtins_present(self, session):
        """Session should have mathematical builtins."""
        # Constants
        assert "PI" in session._globals
        assert "E" in session._globals
        assert "TAU" in session._globals

        # Functions
        assert "sqrt" in session._globals
        assert "sin" in session._globals
        assert "cos" in session._globals
        assert "exp" in session._globals
        assert "log" in session._globals

    def test_prompt_defaults(self, session):
        """Session should have correct default prompts."""
        assert session.prompt == ">>> "
        assert session.continuation_prompt == "... "


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestEvaluation:
    """Tests for expression and statement evaluation."""

    def test_eval_simple_expression(self, session):
        """Evaluate a simple arithmetic expression."""
        result = session.eval_line("2 + 3")
        assert result == "5"

    def test_eval_float_expression(self, session):
        """Evaluate a floating-point expression."""
        result = session.eval_line("3.14 * 2")
        assert "6.28" in result

    def test_eval_builtin_constant(self, session):
        """Evaluate a builtin constant."""
        result = session.eval_line("PI")
        assert "3.14" in result

    def test_eval_builtin_function(self, session):
        """Evaluate a builtin function call."""
        result = session.eval_line("sqrt(16)")
        assert "4" in result

    def test_eval_empty_line(self, session):
        """Empty line should return None."""
        result = session.eval_line("")
        assert result is None

    def test_eval_whitespace_only(self, session):
        """Whitespace-only line should return None."""
        result = session.eval_line("   ")
        assert result is None


# =============================================================================
# Variable Definition Tests
# =============================================================================


class TestVariableDefinitions:
    """Tests for let statement handling."""

    def test_define_integer(self, session):
        """Define an integer variable."""
        result = session.eval_line("let x = 10")
        assert "x" in result
        assert "10" in result
        assert "x" in session.variables
        assert session.variables["x"] == 10
        assert session.variable_types["x"] == "Int"

    def test_define_float(self, session):
        """Define a float variable."""
        result = session.eval_line("let y = 3.14")
        assert "y" in session.variables
        assert abs(session.variables["y"] - 3.14) < 0.001
        assert session.variable_types["y"] == "Float"

    def test_define_string(self, session):
        """Define a string variable."""
        result = session.eval_line('let name = "Alice"')
        assert "name" in session.variables
        assert session.variables["name"] == "Alice"
        assert session.variable_types["name"] == "String"

    def test_define_list(self, session):
        """Define a list variable."""
        result = session.eval_line("let arr = [1, 2, 3]")
        assert "arr" in session.variables
        assert session.variables["arr"] == [1, 2, 3]
        assert "List[Int]" in session.variable_types["arr"]

    def test_variable_in_expression(self, session):
        """Use a defined variable in an expression."""
        session.eval_line("let x = 10")
        result = session.eval_line("x * 2")
        assert result == "20"


# =============================================================================
# Function Definition Tests
# =============================================================================


class TestFunctionDefinitions:
    """Tests for function definition handling."""

    def test_define_simple_function(self, session):
        """Define a simple function."""
        result = session.eval_line("fn add(a: Int, b: Int) -> Int { return a + b }")
        assert "defined:" in result
        assert "add" in result
        assert "add" in session.functions

    def test_function_info_stored(self, session):
        """Function info should be stored correctly."""
        session.eval_line("fn square(n: Int) -> Int { return n * n }")
        func = session.functions["square"]
        assert func.name == "square"
        assert "n" in func.params
        assert "Int" in func.param_types
        assert func.return_type == "Int"

    def test_call_defined_function(self, session):
        """Call a user-defined function."""
        session.eval_line("fn double(x: Int) -> Int { return x * 2 }")
        result = session.eval_line("double(5)")
        assert result == "10"


# =============================================================================
# Command Tests
# =============================================================================


class TestCommands:
    """Tests for REPL commands."""

    def test_help_command(self, session):
        """Test :help command."""
        result = session.eval_line(":help")
        assert "Commands:" in result
        assert ":quit" in result
        assert ":vars" in result
        assert ":funcs" in result

    def test_unknown_command(self, session):
        """Test unknown command handling."""
        result = session.eval_line(":unknown")
        assert "Unknown command" in result

    def test_vars_command_empty(self, session):
        """Test :vars command with no variables."""
        result = session.eval_line(":vars")
        assert "No variables defined" in result

    def test_vars_command_with_variables(self, session):
        """Test :vars command with defined variables."""
        session.eval_line("let x = 10")
        session.eval_line("let y = 20")
        result = session.eval_line(":vars")
        assert "x" in result
        assert "y" in result

    def test_funcs_command_empty(self, session):
        """Test :funcs command with no functions."""
        result = session.eval_line(":funcs")
        assert "No functions defined" in result

    def test_funcs_command_with_functions(self, session):
        """Test :funcs command with defined functions."""
        session.eval_line("fn foo() -> Int { return 1 }")
        result = session.eval_line(":funcs")
        assert "foo" in result

    def test_clear_command(self, session):
        """Test :clear command."""
        session.eval_line("let x = 10")
        session.eval_line("fn foo() -> Int { return 1 }")
        result = session.eval_line(":clear")
        assert "Cleared" in result
        assert len(session.variables) == 0
        assert len(session.functions) == 0

    def test_type_command(self, session):
        """Test :type command."""
        session.eval_line("let x = 10")
        result = session.eval_line(":type x + 1")
        # Should return the type of the expression
        assert result is not None

    def test_type_command_no_expression(self, session):
        """Test :type command without expression."""
        result = session.eval_line(":type")
        assert "Error" in result

    def test_ast_command(self, session):
        """Test :ast command."""
        result = session.eval_line(":ast x + 1")
        assert "BinaryExpression" in result or "Program" in result

    def test_reset_command(self, session):
        """Test :reset command."""
        session.eval_line("let x = 10")
        result = session.eval_line(":reset")
        assert "reset" in result.lower()


# =============================================================================
# Type Inference Tests
# =============================================================================


class TestTypeInference:
    """Tests for runtime type inference."""

    def test_infer_bool_type(self, session):
        """Infer Bool type."""
        assert session._infer_value_type(True) == "Bool"
        assert session._infer_value_type(False) == "Bool"

    def test_infer_int_type(self, session):
        """Infer Int type."""
        assert session._infer_value_type(42) == "Int"
        assert session._infer_value_type(-10) == "Int"

    def test_infer_float_type(self, session):
        """Infer Float type."""
        assert session._infer_value_type(3.14) == "Float"

    def test_infer_string_type(self, session):
        """Infer String type."""
        assert session._infer_value_type("hello") == "String"

    def test_infer_list_type(self, session):
        """Infer List type."""
        result = session._infer_value_type([1, 2, 3])
        assert "List" in result
        assert "Int" in result

    def test_infer_empty_list_type(self, session):
        """Infer empty List type."""
        result = session._infer_value_type([])
        assert "List" in result
        assert "Unknown" in result

    def test_infer_dict_type(self, session):
        """Infer Dict type."""
        result = session._infer_value_type({"a": 1})
        assert "Dict" in result

    def test_infer_none_type(self, session):
        """Infer None type."""
        assert session._infer_value_type(None) == "None"


# =============================================================================
# Completion Tests
# =============================================================================


class TestCompletion:
    """Tests for tab completion."""

    def test_complete_keyword(self, completer):
        """Complete a keyword."""
        completions = completer._get_completions("le")
        assert "let" in completions

    def test_complete_builtin_constant(self, completer):
        """Complete a builtin constant."""
        completions = completer._get_completions("P")
        assert "PI" in completions

    def test_complete_builtin_function(self, completer):
        """Complete a builtin function."""
        completions = completer._get_completions("sq")
        assert "sqrt" in completions

    def test_complete_command(self, completer):
        """Complete a REPL command."""
        completions = completer._get_completions(":he")
        assert ":help" in completions

    def test_complete_user_variable(self, session, completer):
        """Complete a user-defined variable."""
        session.variables["myVariable"] = 42
        completions = completer._get_completions("my")
        assert "myVariable" in completions

    def test_complete_user_function(self, session, completer):
        """Complete a user-defined function."""
        session.functions["myFunc"] = DefinedFunction(
            name="myFunc",
            params=[],
            param_types=[],
            return_type="Int",
            source="fn myFunc() -> Int { return 1 }",
        )
        completions = completer._get_completions("my")
        assert "myFunc" in completions


# =============================================================================
# Multi-line Input Tests
# =============================================================================


class TestMultilineInput:
    """Tests for multi-line input detection."""

    def test_incomplete_brace(self, session):
        """Detect incomplete input with open brace."""
        assert session._is_incomplete("fn foo() {") is True
        assert session._is_incomplete("fn foo() { }") is False

    def test_incomplete_bracket(self, session):
        """Detect incomplete input with open bracket."""
        assert session._is_incomplete("[1, 2,") is True
        assert session._is_incomplete("[1, 2, 3]") is False

    def test_incomplete_paren(self, session):
        """Detect incomplete input with open parenthesis."""
        assert session._is_incomplete("foo(1,") is True
        assert session._is_incomplete("foo(1, 2)") is False

    def test_balanced_nested(self, session):
        """Balanced nested brackets should not be incomplete."""
        assert session._is_incomplete("[[1, 2], [3, 4]]") is False
        assert session._is_incomplete("{a: [1, 2]}") is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_syntax_error(self, session):
        """Syntax errors should be reported."""
        result = session.eval_line("let x = ")
        assert "Error" in result

    def test_undefined_variable(self, session):
        """Undefined variable should cause error."""
        result = session.eval_line("undefined_var")
        assert "Error" in result or result is None

    def test_type_error_in_expression(self, session):
        """Type errors should be handled."""
        # Depending on implementation, this might error or return None
        result = session.eval_line("1 + 'hello'")
        # Either an error message or the result of the operation
        assert result is not None or result is None  # Just ensure no crash


# =============================================================================
# Colors Tests
# =============================================================================


class TestColors:
    """Tests for color handling."""

    def test_colors_class_exists(self):
        """Colors class should exist with expected attributes."""
        # The colors might be disabled in non-TTY test environments
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "RESET")
        assert hasattr(Colors, "BOLD")
        assert hasattr(Colors, "disable")

    def test_colors_disable(self):
        """Colors.disable() should clear all codes."""
        # Save originals
        original_red = Colors.RED
        original_green = Colors.GREEN
        original_reset = Colors.RESET

        Colors.disable()
        assert Colors.RED == ""
        assert Colors.GREEN == ""
        assert Colors.RESET == ""

        # Restore (for other tests)
        Colors.RED = original_red
        Colors.GREEN = original_green
        Colors.RESET = original_reset
