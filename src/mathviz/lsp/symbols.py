"""
Symbol table and scope management for MathViz LSP.

This module provides symbol tracking, scope management, and symbol lookup
functionality for the language server. It collects information about all
definitions (functions, variables, types, etc.) in a document.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from lsprotocol import types


class SymbolKind(Enum):
    """Kind of symbol in the MathViz language."""

    FUNCTION = auto()
    VARIABLE = auto()
    CONSTANT = auto()
    PARAMETER = auto()
    TYPE = auto()
    CLASS = auto()
    STRUCT = auto()
    ENUM = auto()
    ENUM_VARIANT = auto()
    TRAIT = auto()
    METHOD = auto()
    FIELD = auto()
    MODULE = auto()
    SCENE = auto()
    IMPORT = auto()


# Map MathViz symbol kinds to LSP symbol kinds
SYMBOL_KIND_TO_LSP: dict[SymbolKind, types.SymbolKind] = {
    SymbolKind.FUNCTION: types.SymbolKind.Function,
    SymbolKind.VARIABLE: types.SymbolKind.Variable,
    SymbolKind.CONSTANT: types.SymbolKind.Constant,
    SymbolKind.PARAMETER: types.SymbolKind.Variable,
    SymbolKind.TYPE: types.SymbolKind.TypeParameter,
    SymbolKind.CLASS: types.SymbolKind.Class,
    SymbolKind.STRUCT: types.SymbolKind.Struct,
    SymbolKind.ENUM: types.SymbolKind.Enum,
    SymbolKind.ENUM_VARIANT: types.SymbolKind.EnumMember,
    SymbolKind.TRAIT: types.SymbolKind.Interface,
    SymbolKind.METHOD: types.SymbolKind.Method,
    SymbolKind.FIELD: types.SymbolKind.Field,
    SymbolKind.MODULE: types.SymbolKind.Module,
    SymbolKind.SCENE: types.SymbolKind.Class,
    SymbolKind.IMPORT: types.SymbolKind.Module,
}


@dataclass
class Location:
    """Source location information for a symbol."""

    uri: str
    line: int  # 0-indexed
    character: int  # 0-indexed
    end_line: int | None = None
    end_character: int | None = None

    def to_lsp_location(self) -> types.Location:
        """Convert to LSP Location type."""
        return types.Location(
            uri=self.uri,
            range=types.Range(
                start=types.Position(line=self.line, character=self.character),
                end=types.Position(
                    line=self.end_line if self.end_line is not None else self.line,
                    character=self.end_character
                    if self.end_character is not None
                    else self.character + 1,
                ),
            ),
        )

    def to_lsp_range(self) -> types.Range:
        """Convert to LSP Range type."""
        return types.Range(
            start=types.Position(line=self.line, character=self.character),
            end=types.Position(
                line=self.end_line if self.end_line is not None else self.line,
                character=self.end_character
                if self.end_character is not None
                else self.character + 1,
            ),
        )


@dataclass
class Symbol:
    """
    Represents a symbol (function, variable, type, etc.) in the source code.

    Attributes:
        name: The symbol's identifier name
        kind: The kind of symbol (function, variable, etc.)
        type_info: Type information as a string (e.g., "Int", "(Int, Int) -> Float")
        doc: Documentation string if available
        location: Where the symbol is defined
        scope: The scope path where this symbol is defined (e.g., "MyClass.my_method")
        signature: Full signature for functions/methods
        children: Child symbols (fields, methods, variants)
    """

    name: str
    kind: SymbolKind
    type_info: str | None = None
    doc: str | None = None
    location: Location | None = None
    scope: str = ""
    signature: str | None = None
    children: list["Symbol"] = field(default_factory=list)

    def to_lsp_symbol_kind(self) -> types.SymbolKind:
        """Get the LSP symbol kind."""
        return SYMBOL_KIND_TO_LSP.get(self.kind, types.SymbolKind.Variable)

    def to_document_symbol(self) -> types.DocumentSymbol:
        """Convert to LSP DocumentSymbol."""
        if self.location is None:
            # Default location if none provided
            range_ = types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=len(self.name)),
            )
        else:
            range_ = self.location.to_lsp_range()

        children = [child.to_document_symbol() for child in self.children]

        return types.DocumentSymbol(
            name=self.name,
            kind=self.to_lsp_symbol_kind(),
            range=range_,
            selection_range=range_,
            detail=self.type_info,
            children=children if children else None,
        )


class SymbolTable:
    """
    Symbol table for tracking definitions across a document.

    Supports nested scopes and efficient symbol lookup.
    """

    def __init__(self) -> None:
        """Initialize an empty symbol table."""
        # All symbols by name (name -> list of symbols with that name)
        self.symbols: dict[str, list[Symbol]] = {}

        # Symbols organized by scope
        self.scopes: dict[str, list[Symbol]] = {"": []}

        # Current scope stack for nested scope tracking
        self._scope_stack: list[str] = [""]

        # References to symbols (for find-references)
        # Maps (symbol_name, definition_location) -> list of reference locations
        self.references: dict[tuple[str, str], list[Location]] = {}

    @property
    def current_scope(self) -> str:
        """Get the current scope path."""
        return self._scope_stack[-1] if self._scope_stack else ""

    def enter_scope(self, name: str) -> None:
        """
        Enter a new nested scope.

        Args:
            name: The name of the new scope (e.g., function name, class name)
        """
        parent = self.current_scope
        new_scope = f"{parent}.{name}" if parent else name
        self._scope_stack.append(new_scope)

        if new_scope not in self.scopes:
            self.scopes[new_scope] = []

    def exit_scope(self) -> None:
        """Exit the current scope, returning to the parent scope."""
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()

    def add_symbol(self, symbol: Symbol) -> None:
        """
        Add a symbol to the table.

        Args:
            symbol: The symbol to add
        """
        # Update scope if not explicitly set
        if not symbol.scope:
            symbol.scope = self.current_scope

        # Add to name-based lookup
        if symbol.name not in self.symbols:
            self.symbols[symbol.name] = []
        self.symbols[symbol.name].append(symbol)

        # Add to scope-based organization
        scope = symbol.scope
        if scope not in self.scopes:
            self.scopes[scope] = []
        self.scopes[scope].append(symbol)

    def add_reference(self, name: str, def_location: Location, ref_location: Location) -> None:
        """
        Track a reference to a symbol.

        Args:
            name: The symbol name being referenced
            def_location: The location of the symbol's definition
            ref_location: The location of this reference
        """
        key = (name, f"{def_location.uri}:{def_location.line}:{def_location.character}")
        if key not in self.references:
            self.references[key] = []
        self.references[key].append(ref_location)

    def lookup(
        self,
        name: str,
        scope: str | None = None,
        include_parent_scopes: bool = True,
    ) -> Symbol | None:
        """
        Look up a symbol by name.

        Args:
            name: The symbol name to look up
            scope: The scope to search in (defaults to current scope)
            include_parent_scopes: Whether to search parent scopes

        Returns:
            The matching symbol, or None if not found
        """
        if scope is None:
            scope = self.current_scope

        # Check the specified scope
        if scope in self.scopes:
            for symbol in self.scopes[scope]:
                if symbol.name == name:
                    return symbol

        # Check parent scopes if requested
        if include_parent_scopes and scope:
            parent_scope = ".".join(scope.split(".")[:-1]) if "." in scope else ""
            return self.lookup(name, parent_scope, include_parent_scopes=True)

        # Check global scope
        if scope != "" and "" in self.scopes:
            for symbol in self.scopes[""]:
                if symbol.name == name:
                    return symbol

        return None

    def lookup_all(self, name: str) -> list[Symbol]:
        """
        Get all symbols with a given name.

        Args:
            name: The symbol name to look up

        Returns:
            List of all matching symbols
        """
        return self.symbols.get(name, [])

    def get_completions_in_scope(self, scope: str | None = None) -> list[Symbol]:
        """
        Get all symbols visible from a given scope.

        Args:
            scope: The scope to get completions for (defaults to current)

        Returns:
            List of visible symbols
        """
        if scope is None:
            scope = self.current_scope

        result: list[Symbol] = []
        seen_names: set[str] = set()

        # Collect symbols from current scope up to global
        current = scope
        while True:
            if current in self.scopes:
                for symbol in self.scopes[current]:
                    if symbol.name not in seen_names:
                        result.append(symbol)
                        seen_names.add(symbol.name)

            if not current:
                break

            # Move to parent scope
            current = ".".join(current.split(".")[:-1]) if "." in current else ""

        return result

    def get_all_symbols(self) -> list[Symbol]:
        """Get all symbols in the table."""
        result: list[Symbol] = []
        for symbols in self.symbols.values():
            result.extend(symbols)
        return result

    def get_references(self, symbol: Symbol) -> list[Location]:
        """
        Get all references to a symbol.

        Args:
            symbol: The symbol to find references for

        Returns:
            List of reference locations
        """
        if symbol.location is None:
            return []

        key = (
            symbol.name,
            f"{symbol.location.uri}:{symbol.location.line}:{symbol.location.character}",
        )
        return self.references.get(key, [])

    def get_symbol_at_position(self, line: int, character: int) -> Symbol | None:
        """
        Find the symbol at a given position.

        Args:
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            The symbol at that position, or None
        """
        for symbols in self.symbols.values():
            for symbol in symbols:
                if symbol.location is None:
                    continue

                loc = symbol.location
                # Check if position is within symbol's location range
                if loc.line <= line <= (loc.end_line or loc.line):
                    start_char = loc.character if line == loc.line else 0
                    end_char = (
                        (loc.end_character or loc.character + len(symbol.name))
                        if line == (loc.end_line or loc.line)
                        else float("inf")
                    )
                    if start_char <= character < end_char:
                        return symbol

        return None

    def clear(self) -> None:
        """Clear all symbols from the table."""
        self.symbols.clear()
        self.scopes = {"": []}
        self._scope_stack = [""]
        self.references.clear()
