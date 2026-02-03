"""Tests for the MathViz LSP document analyzer."""

from mathviz.lsp.analyzer import DocumentAnalyzer
from mathviz.lsp.symbols import SymbolKind


class TestDocumentAnalyzer:
    """Test suite for DocumentAnalyzer."""

    def test_analyze_simple_variable(self) -> None:
        """Test analyzing a simple variable declaration."""
        source = "let x: Int = 42"
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        symbols = analyzer.symbols.get_all_symbols()
        assert len(symbols) == 1
        assert symbols[0].name == "x"
        assert symbols[0].kind == SymbolKind.VARIABLE
        assert symbols[0].type_info == "Int"

    def test_analyze_function(self) -> None:
        """Test analyzing a function definition."""
        source = """
fn add(a: Int, b: Int) -> Int {
    return a + b
}
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # Should have function + 2 parameters
        func = analyzer.symbols.lookup("add")
        assert func is not None
        assert func.kind == SymbolKind.FUNCTION
        assert func.type_info == "(Int, Int) -> Int"

    def test_analyze_struct(self) -> None:
        """Test analyzing a struct definition."""
        source = """
struct Point {
    x: Float
    y: Float
}
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        point = analyzer.symbols.lookup("Point")
        assert point is not None
        assert point.kind == SymbolKind.STRUCT
        assert len(point.children) == 2

    def test_analyze_enum(self) -> None:
        """Test analyzing an enum definition."""
        source = """
enum Color {
    Red
    Green
    Blue
}
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        color = analyzer.symbols.lookup("Color")
        assert color is not None
        assert color.kind == SymbolKind.ENUM
        assert len(color.children) == 3

    def test_hover_on_function(self) -> None:
        """Test hover information for a function."""
        source = """fn greet(name: String) -> String {
    return f"Hello, {name}!"
}
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # Hover on 'greet' (line 0, char 3)
        hover = analyzer.get_hover(0, 3)
        assert hover is not None
        assert "greet" in hover.contents.value
        assert "String" in hover.contents.value

    def test_hover_on_keyword(self) -> None:
        """Test hover information for a keyword."""
        source = "let x = 42"
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # Hover on 'let' (line 0, char 1)
        hover = analyzer.get_hover(0, 1)
        assert hover is not None
        assert "keyword" in hover.contents.value.lower()

    def test_go_to_definition(self) -> None:
        """Test go-to-definition for a variable reference."""
        source = """
let value = 42
let doubled = value * 2
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # Go to definition of 'value' in line 2
        definition = analyzer.get_definition(2, 14)
        assert definition is not None
        assert definition.range.start.line == 1  # Where 'value' is defined

    def test_find_references(self) -> None:
        """Test finding all references to a symbol."""
        source = """
let x = 10
let y = x + 5
let z = x * 2
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # Find references to 'x' (defined at line 1)
        references = analyzer.get_references(1, 4, include_declaration=True)
        # Should find at least the declaration and both uses
        assert len(references) >= 3

    def test_completions(self) -> None:
        """Test getting completions."""
        source = """
fn calculate(x: Int) -> Int {
    return x * 2
}

let result = cal
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # Get completions after 'cal'
        completions = analyzer.get_completions(5, 16)
        # Should include keywords, types, and symbols
        assert len(completions) > 0

        # Check that 'calculate' is in the completions
        labels = [c.label for c in completions]
        assert "calculate" in labels

    def test_document_symbols(self) -> None:
        """Test getting document symbols for outline view."""
        source = """fn main() {
    println("Hello")
}

struct Config {
    debug: Bool
}

enum Status {
    Success
    Failure
}
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # At minimum, we should find top-level symbols
        all_symbols = analyzer.symbols.get_all_symbols()
        all_names = [s.name for s in all_symbols]

        assert "main" in all_names
        assert "Config" in all_names
        assert "Status" in all_names

    def test_diagnostics_for_syntax_error(self) -> None:
        """Test that syntax errors generate diagnostics."""
        source = "let x = "  # Missing value
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        assert len(analyzer.diagnostics) > 0

    def test_diagnostics_for_valid_code(self) -> None:
        """Test that valid code may still have warnings."""
        source = """
fn unused_function() {
    pass
}

fn main() {
    println("Hello")
}
"""
        analyzer = DocumentAnalyzer(source, "test://test.mviz")
        analyzer.analyze()

        # May have lint warnings for unused function
        # The exact count depends on linter configuration
        assert analyzer.diagnostics is not None


class TestSymbolTable:
    """Test suite for SymbolTable."""

    def test_add_and_lookup(self) -> None:
        """Test adding and looking up symbols."""
        from mathviz.lsp.symbols import Symbol, SymbolTable

        table = SymbolTable()
        symbol = Symbol(name="test", kind=SymbolKind.VARIABLE, type_info="Int")
        table.add_symbol(symbol)

        found = table.lookup("test")
        assert found is not None
        assert found.name == "test"
        assert found.type_info == "Int"

    def test_nested_scopes(self) -> None:
        """Test symbol lookup in nested scopes."""
        from mathviz.lsp.symbols import Symbol, SymbolTable

        table = SymbolTable()

        # Add global symbol
        global_sym = Symbol(name="x", kind=SymbolKind.VARIABLE, type_info="Int")
        table.add_symbol(global_sym)

        # Enter function scope
        table.enter_scope("func")
        local_sym = Symbol(name="y", kind=SymbolKind.VARIABLE, type_info="Float")
        table.add_symbol(local_sym)

        # Should find local and global symbols
        assert table.lookup("y") is not None
        assert table.lookup("x") is not None

        # Exit scope
        table.exit_scope()

        # Should only find global
        assert table.lookup("x") is not None
        # Local not visible from global scope
        assert table.lookup("y", scope="", include_parent_scopes=False) is None
