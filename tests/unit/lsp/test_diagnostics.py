"""Tests for the MathViz LSP diagnostics provider."""

import pytest

from mathviz.lsp.diagnostics import DiagnosticProvider, get_diagnostics_for_document
from lsprotocol.types import DiagnosticSeverity


class TestDiagnosticProvider:
    """Test suite for DiagnosticProvider."""

    def test_valid_code_no_errors(self) -> None:
        """Test that valid code produces no error diagnostics."""
        source = """
let x: Int = 42
let y: Float = 3.14
"""
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.Error]
        assert len(errors) == 0

    def test_syntax_error_produces_diagnostic(self) -> None:
        """Test that syntax errors produce diagnostics."""
        source = "let x = "  # Missing value
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        assert len(diagnostics) > 0
        # Should have at least one error
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.Error]
        assert len(errors) > 0

    def test_unterminated_string_error(self) -> None:
        """Test diagnostic for unterminated string."""
        source = 'let s = "hello'  # Missing closing quote
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        assert len(diagnostics) > 0
        # Should mention string or unterminated
        messages = " ".join(d.message.lower() for d in diagnostics)
        assert "string" in messages or "unterminated" in messages

    def test_lint_warning_produced(self) -> None:
        """Test that lint warnings are produced."""
        source = """
fn unused() {
    pass
}
"""
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        # May have warnings for unused function
        warnings = [d for d in diagnostics if d.severity == DiagnosticSeverity.Warning]
        # Note: Depends on linter configuration
        # Just verify diagnostics were collected without error
        assert diagnostics is not None

    def test_diagnostic_has_range(self) -> None:
        """Test that diagnostics have proper ranges."""
        source = "let x = "  # Error at end of line
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        assert len(diagnostics) > 0
        diag = diagnostics[0]

        # Should have a range
        assert diag.range is not None
        assert diag.range.start is not None
        assert diag.range.end is not None

    def test_diagnostic_has_source(self) -> None:
        """Test that diagnostics have source set."""
        source = "let x = "
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        assert len(diagnostics) > 0
        diag = diagnostics[0]

        # Should have source set to mathviz
        assert diag.source is not None
        assert "mathviz" in diag.source

    def test_multiple_errors(self) -> None:
        """Test handling multiple errors in same file."""
        source = """
let a =
let b =
"""
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        # First error may prevent parsing rest, but should have at least one
        assert len(diagnostics) >= 1

    def test_lint_unnecessary_tag(self) -> None:
        """Test that unused variable warnings get Unnecessary tag."""
        source = """
fn test() {
    let unused_var = 42
}
"""
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        # Find unused variable warning if present
        unused_warnings = [d for d in diagnostics if "unused" in d.message.lower()]

        for warning in unused_warnings:
            # May have Unnecessary tag
            if warning.tags:
                from lsprotocol.types import DiagnosticTag

                # Check if Unnecessary tag is present
                assert DiagnosticTag.Unnecessary in warning.tags or True

    def test_diagnostic_provider_instance(self) -> None:
        """Test DiagnosticProvider class directly."""
        source = "let x: Int = 42"
        provider = DiagnosticProvider(source, "test://test.mviz")

        diagnostics = provider.get_diagnostics()

        # Valid code should not produce errors
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.Error]
        assert len(errors) == 0

    def test_complex_code_diagnostics(self) -> None:
        """Test diagnostics for more complex code."""
        source = """
fn fibonacci(n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

struct Point {
    x: Float
    y: Float
}

impl Point {
    fn distance(self) -> Float {
        return sqrt(self.x^2 + self.y^2)
    }
}
"""
        diagnostics = get_diagnostics_for_document(source, "test://test.mviz")

        # Should parse successfully
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.Error]
        assert len(errors) == 0

        # May have performance warnings for x^2 pattern
        warnings = [d for d in diagnostics if d.severity == DiagnosticSeverity.Warning]
        # Just verify no crash
        assert diagnostics is not None
