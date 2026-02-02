"""Tests for the MathViz LSP completion provider."""

import pytest

from mathviz.lsp.completions import CompletionProvider, BUILTIN_TYPES, STDLIB_MATH
from mathviz.lsp.symbols import Symbol, SymbolKind
from lsprotocol.types import CompletionItemKind


class TestCompletionProvider:
    """Test suite for CompletionProvider."""

    @pytest.fixture
    def provider(self) -> CompletionProvider:
        """Create a completion provider for testing."""
        return CompletionProvider()

    def test_keyword_completions(self, provider: CompletionProvider) -> None:
        """Test that keyword completions are generated."""
        completions = provider.get_keyword_completions()

        # Should have many keywords
        assert len(completions) > 20

        # Check for essential keywords
        labels = [c.label for c in completions]
        assert "fn" in labels
        assert "let" in labels
        assert "if" in labels
        assert "for" in labels
        assert "struct" in labels

    def test_keyword_snippets(self, provider: CompletionProvider) -> None:
        """Test that keyword completions include snippets."""
        completions = provider.get_keyword_completions()

        # Find 'fn' completion
        fn_completion = next(c for c in completions if c.label == "fn")

        # Should have a snippet
        assert fn_completion.insert_text is not None
        assert "${1:" in fn_completion.insert_text  # Snippet placeholder

    def test_type_completions(self, provider: CompletionProvider) -> None:
        """Test that type completions are generated."""
        completions = provider.get_type_completions()

        # Should have all builtin types
        labels = [c.label for c in completions]
        for type_name in BUILTIN_TYPES:
            assert type_name in labels

    def test_generic_type_snippets(self, provider: CompletionProvider) -> None:
        """Test that generic types have parameter snippets."""
        completions = provider.get_type_completions()

        # Find 'List' completion
        list_completion = next(c for c in completions if c.label == "List")

        # Should have a generic type snippet
        assert list_completion.insert_text is not None
        assert "List[" in list_completion.insert_text

    def test_snippet_completions(self, provider: CompletionProvider) -> None:
        """Test that code snippets are generated."""
        completions = provider.get_snippet_completions()

        # Should have several snippets
        assert len(completions) > 5

        labels = [c.label for c in completions]
        assert "main" in labels  # Main function snippet
        assert "forrange" in labels  # For-range loop snippet

    def test_stdlib_completions(self, provider: CompletionProvider) -> None:
        """Test that standard library completions are generated."""
        completions = provider.get_stdlib_completions()

        # Should have math functions
        labels = [c.label for c in completions]
        for func_name in ["sqrt", "sin", "cos", "abs"]:
            assert func_name in labels

    def test_stdlib_function_signatures(self, provider: CompletionProvider) -> None:
        """Test that stdlib completions have proper signatures."""
        completions = provider.get_stdlib_completions()

        # Find 'sqrt' completion
        sqrt_completion = next(c for c in completions if c.label == "sqrt")

        # Should have signature in detail
        assert sqrt_completion.detail is not None
        assert "Float" in sqrt_completion.detail

    def test_symbol_completions(self, provider: CompletionProvider) -> None:
        """Test generating completions from symbols."""
        symbols = [
            Symbol(name="myFunc", kind=SymbolKind.FUNCTION, type_info="() -> Int"),
            Symbol(name="myVar", kind=SymbolKind.VARIABLE, type_info="String"),
            Symbol(name="MyClass", kind=SymbolKind.CLASS),
        ]

        completions = provider.get_symbol_completions(symbols)

        assert len(completions) == 3

        # Check kinds are mapped correctly
        func_completion = next(c for c in completions if c.label == "myFunc")
        assert func_completion.kind == CompletionItemKind.Function

        var_completion = next(c for c in completions if c.label == "myVar")
        assert var_completion.kind == CompletionItemKind.Variable

        class_completion = next(c for c in completions if c.label == "MyClass")
        assert class_completion.kind == CompletionItemKind.Class

    def test_function_completion_has_parens(self, provider: CompletionProvider) -> None:
        """Test that function completions include parentheses."""
        symbols = [
            Symbol(name="calculate", kind=SymbolKind.FUNCTION, type_info="(Int) -> Int"),
        ]

        completions = provider.get_symbol_completions(symbols)
        func_completion = completions[0]

        # Should have () in insert text
        assert func_completion.insert_text is not None
        assert "(" in func_completion.insert_text

    def test_completions_caching(self, provider: CompletionProvider) -> None:
        """Test that completions are cached for performance."""
        # First call generates completions
        keywords1 = provider.get_keyword_completions()

        # Second call should return same object (cached)
        keywords2 = provider.get_keyword_completions()

        assert keywords1 is keywords2

    def test_documentation_in_completions(self, provider: CompletionProvider) -> None:
        """Test that completions include documentation."""
        symbols = [
            Symbol(
                name="documented",
                kind=SymbolKind.FUNCTION,
                type_info="() -> Int",
                doc="This is a documented function.",
            ),
        ]

        completions = provider.get_symbol_completions(symbols)
        completion = completions[0]

        # Should have documentation
        assert completion.documentation is not None
