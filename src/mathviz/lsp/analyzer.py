"""
Document analysis for MathViz LSP.

This module provides comprehensive document analysis including:
- Parsing and AST generation
- Symbol collection
- Type inference
- Position-based queries (hover, go-to-definition, etc.)
"""

from dataclasses import dataclass

from lsprotocol import types

from mathviz.compiler.ast_nodes import (
    Block,
    ClassDef,
    ConstDeclaration,
    EnumDef,
    ForStatement,
    FunctionDef,
    GenericType,
    IfStatement,
    ImplBlock,
    ImportStatement,
    LetStatement,
    Method,
    Program,
    SceneDef,
    SimpleType,
    Statement,
    StructDef,
    TraitDef,
    TraitMethod,
    TypeAnnotation,
    UseStatement,
    WhileStatement,
)
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser
from mathviz.compiler.tokens import Token
from mathviz.lsp.completions import CompletionContext, CompletionProvider
from mathviz.lsp.diagnostics import get_diagnostics_for_document
from mathviz.lsp.symbols import Location, Symbol, SymbolKind, SymbolTable


@dataclass
class HoverInfo:
    """Information for hover display."""

    content: str
    range: types.Range | None = None


@dataclass
class DefinitionInfo:
    """Information for go-to-definition."""

    uri: str
    range: types.Range


class DocumentAnalyzer:
    """
    Analyzes a MathViz document for LSP features.

    This class parses the document, collects symbols, and provides
    position-based queries for hover, completions, definitions, etc.
    """

    def __init__(self, source: str, uri: str) -> None:
        """
        Initialize the analyzer with source code.

        Args:
            source: The MathViz source code
            uri: The document URI
        """
        self.source = source
        self.uri = uri
        self.lines = source.splitlines()

        # Analysis results
        self.tokens: list[Token] = []
        self.ast: Program | None = None
        self.symbols = SymbolTable()
        self.diagnostics: list[types.Diagnostic] = []

        # Parsing state
        self._parse_error: str | None = None

        # Completion provider
        self._completion_provider = CompletionProvider()

    def analyze(self) -> None:
        """
        Parse and analyze the document.

        This method:
        1. Tokenizes the source
        2. Parses into an AST
        3. Collects symbols
        4. Generates diagnostics
        """
        # Clear previous state
        self.symbols.clear()
        self.diagnostics = []
        self._parse_error = None

        # Phase 1: Tokenize
        try:
            lexer = Lexer(self.source, filename=self.uri)
            self.tokens = lexer.tokenize()
        except Exception as e:
            self._parse_error = str(e)
            self.diagnostics = get_diagnostics_for_document(self.source, self.uri)
            return

        # Phase 2: Parse
        try:
            parser = Parser(self.tokens, source=self.source, filename=self.uri)
            self.ast = parser.parse()
        except Exception as e:
            self._parse_error = str(e)
            self.diagnostics = get_diagnostics_for_document(self.source, self.uri)
            return

        # Phase 3: Collect symbols
        if self.ast:
            self._collect_symbols(self.ast)

        # Phase 4: Get diagnostics
        self.diagnostics = get_diagnostics_for_document(self.source, self.uri)

    def _collect_symbols(self, program: Program) -> None:
        """
        Collect all symbols from the AST.

        Args:
            program: The parsed program AST
        """
        for stmt in program.statements:
            self._collect_statement_symbols(stmt)

    def _collect_statement_symbols(self, stmt: Statement) -> None:
        """Collect symbols from a statement."""
        if isinstance(stmt, FunctionDef):
            self._collect_function_symbols(stmt)
        elif isinstance(stmt, LetStatement):
            self._collect_let_symbols(stmt)
        elif isinstance(stmt, ConstDeclaration):
            self._collect_const_symbols(stmt)
        elif isinstance(stmt, ClassDef):
            self._collect_class_symbols(stmt)
        elif isinstance(stmt, SceneDef):
            self._collect_scene_symbols(stmt)
        elif isinstance(stmt, StructDef):
            self._collect_struct_symbols(stmt)
        elif isinstance(stmt, EnumDef):
            self._collect_enum_symbols(stmt)
        elif isinstance(stmt, TraitDef):
            self._collect_trait_symbols(stmt)
        elif isinstance(stmt, ImplBlock):
            self._collect_impl_symbols(stmt)
        elif isinstance(stmt, ImportStatement):
            self._collect_import_symbols(stmt)
        elif isinstance(stmt, UseStatement):
            self._collect_use_symbols(stmt)
        elif isinstance(stmt, ForStatement):
            self._collect_for_symbols(stmt)
        elif isinstance(stmt, IfStatement):
            self._collect_if_symbols(stmt)
        elif isinstance(stmt, WhileStatement):
            self._collect_while_symbols(stmt)

    def _collect_function_symbols(self, func: FunctionDef) -> None:
        """Collect symbols from a function definition."""
        # Build type signature
        param_types = []
        for param in func.parameters:
            if param.type_annotation:
                param_types.append(self._format_type(param.type_annotation))
            else:
                param_types.append("?")

        return_type = self._format_type(func.return_type) if func.return_type else "?"
        signature = f"fn {func.name}({', '.join(param_types)}) -> {return_type}"
        type_info = f"({', '.join(param_types)}) -> {return_type}"

        location = self._make_location(func.location) if func.location else None

        # Get doc comment
        doc = func.doc_comment.content if func.doc_comment else None

        symbol = Symbol(
            name=func.name,
            kind=SymbolKind.FUNCTION,
            type_info=type_info,
            doc=doc,
            location=location,
            signature=signature,
        )
        self.symbols.add_symbol(symbol)

        # Enter function scope for parameters and body
        self.symbols.enter_scope(func.name)

        # Add parameters as symbols
        for param in func.parameters:
            param_type = self._format_type(param.type_annotation) if param.type_annotation else None
            param_loc = self._make_location(param.location) if param.location else None
            param_symbol = Symbol(
                name=param.name,
                kind=SymbolKind.PARAMETER,
                type_info=param_type,
                location=param_loc,
            )
            self.symbols.add_symbol(param_symbol)

        # Collect symbols from function body
        if func.body:
            self._collect_block_symbols(func.body)

        self.symbols.exit_scope()

    def _collect_let_symbols(self, stmt: LetStatement) -> None:
        """Collect symbols from a let statement."""
        type_info = self._format_type(stmt.type_annotation) if stmt.type_annotation else None
        location = self._make_location(stmt.location) if stmt.location else None

        symbol = Symbol(
            name=stmt.name,
            kind=SymbolKind.VARIABLE,
            type_info=type_info,
            location=location,
        )
        self.symbols.add_symbol(symbol)

    def _collect_const_symbols(self, stmt: ConstDeclaration) -> None:
        """Collect symbols from a const declaration."""
        type_info = self._format_type(stmt.type_annotation) if stmt.type_annotation else None
        location = self._make_location(stmt.location) if stmt.location else None

        symbol = Symbol(
            name=stmt.name,
            kind=SymbolKind.CONSTANT,
            type_info=type_info,
            location=location,
        )
        self.symbols.add_symbol(symbol)

    def _collect_class_symbols(self, cls: ClassDef) -> None:
        """Collect symbols from a class definition."""
        location = self._make_location(cls.location) if cls.location else None
        doc = cls.doc_comment.content if cls.doc_comment else None

        symbol = Symbol(
            name=cls.name,
            kind=SymbolKind.CLASS,
            type_info=cls.name,
            doc=doc,
            location=location,
        )
        self.symbols.add_symbol(symbol)

        # Enter class scope
        self.symbols.enter_scope(cls.name)

        if cls.body:
            self._collect_block_symbols(cls.body)

        self.symbols.exit_scope()

    def _collect_scene_symbols(self, scene: SceneDef) -> None:
        """Collect symbols from a scene definition."""
        location = self._make_location(scene.location) if scene.location else None

        symbol = Symbol(
            name=scene.name,
            kind=SymbolKind.SCENE,
            type_info="Scene",
            location=location,
        )
        self.symbols.add_symbol(symbol)

        # Enter scene scope
        self.symbols.enter_scope(scene.name)

        if scene.body:
            self._collect_block_symbols(scene.body)

        self.symbols.exit_scope()

    def _collect_struct_symbols(self, struct: StructDef) -> None:
        """Collect symbols from a struct definition."""
        location = self._make_location(struct.location) if struct.location else None
        doc = struct.doc_comment.content if struct.doc_comment else None

        # Collect field children
        children: list[Symbol] = []
        for field in struct.fields:
            field_type = self._format_type(field.type_annotation)
            field_loc = self._make_location(field.location) if field.location else None
            field_symbol = Symbol(
                name=field.name,
                kind=SymbolKind.FIELD,
                type_info=field_type,
                location=field_loc,
            )
            children.append(field_symbol)

        symbol = Symbol(
            name=struct.name,
            kind=SymbolKind.STRUCT,
            type_info=struct.name,
            doc=doc,
            location=location,
            children=children,
        )
        self.symbols.add_symbol(symbol)

        # Also add fields at struct scope
        self.symbols.enter_scope(struct.name)
        for child in children:
            self.symbols.add_symbol(child)
        self.symbols.exit_scope()

    def _collect_enum_symbols(self, enum: EnumDef) -> None:
        """Collect symbols from an enum definition."""
        location = self._make_location(enum.location) if enum.location else None
        doc = enum.doc_comment.content if enum.doc_comment else None

        # Collect variant children
        children: list[Symbol] = []
        for variant in enum.variants:
            variant_loc = self._make_location(variant.location) if variant.location else None
            variant_symbol = Symbol(
                name=variant.name,
                kind=SymbolKind.ENUM_VARIANT,
                type_info=enum.name,
                location=variant_loc,
            )
            children.append(variant_symbol)

        symbol = Symbol(
            name=enum.name,
            kind=SymbolKind.ENUM,
            type_info=enum.name,
            doc=doc,
            location=location,
            children=children,
        )
        self.symbols.add_symbol(symbol)

        # Also add variants at enum scope
        self.symbols.enter_scope(enum.name)
        for child in children:
            self.symbols.add_symbol(child)
        self.symbols.exit_scope()

    def _collect_trait_symbols(self, trait: TraitDef) -> None:
        """Collect symbols from a trait definition."""
        location = self._make_location(trait.location) if trait.location else None
        doc = trait.doc_comment.content if trait.doc_comment else None

        symbol = Symbol(
            name=trait.name,
            kind=SymbolKind.TRAIT,
            type_info=trait.name,
            doc=doc,
            location=location,
        )
        self.symbols.add_symbol(symbol)

        # Enter trait scope for methods
        self.symbols.enter_scope(trait.name)
        for method in trait.methods:
            self._collect_trait_method_symbols(method)
        self.symbols.exit_scope()

    def _collect_trait_method_symbols(self, method: TraitMethod) -> None:
        """Collect symbols from a trait method."""
        param_types = []
        for param in method.parameters:
            if param.type_annotation:
                param_types.append(self._format_type(param.type_annotation))
            else:
                param_types.append("?")

        return_type = self._format_type(method.return_type) if method.return_type else "()"
        signature = f"fn {method.name}({', '.join(param_types)}) -> {return_type}"

        location = self._make_location(method.location) if method.location else None

        symbol = Symbol(
            name=method.name,
            kind=SymbolKind.METHOD,
            type_info=f"({', '.join(param_types)}) -> {return_type}",
            signature=signature,
            location=location,
        )
        self.symbols.add_symbol(symbol)

    def _collect_impl_symbols(self, impl: ImplBlock) -> None:
        """Collect symbols from an impl block."""
        scope_name = impl.target_type
        if impl.trait_name:
            scope_name = f"{impl.trait_name}_{impl.target_type}"

        self.symbols.enter_scope(scope_name)
        for method in impl.methods:
            self._collect_method_symbols(method)
        self.symbols.exit_scope()

    def _collect_method_symbols(self, method: Method) -> None:
        """Collect symbols from a method."""
        param_types = []
        if method.has_self:
            param_types.append("self")
        for param in method.parameters:
            if param.type_annotation:
                param_types.append(self._format_type(param.type_annotation))
            else:
                param_types.append("?")

        return_type = self._format_type(method.return_type) if method.return_type else "()"
        signature = f"fn {method.name}({', '.join(param_types)}) -> {return_type}"

        location = self._make_location(method.location) if method.location else None

        symbol = Symbol(
            name=method.name,
            kind=SymbolKind.METHOD,
            type_info=f"({', '.join(param_types)}) -> {return_type}",
            signature=signature,
            location=location,
        )
        self.symbols.add_symbol(symbol)

        # Enter method scope for body
        self.symbols.enter_scope(method.name)
        if method.body:
            self._collect_block_symbols(method.body)
        self.symbols.exit_scope()

    def _collect_import_symbols(self, stmt: ImportStatement) -> None:
        """Collect symbols from an import statement."""
        location = self._make_location(stmt.location) if stmt.location else None

        if stmt.is_from_import:
            for name, alias in stmt.names:
                symbol_name = alias if alias else name
                symbol = Symbol(
                    name=symbol_name,
                    kind=SymbolKind.IMPORT,
                    type_info=f"from {stmt.module}",
                    location=location,
                )
                self.symbols.add_symbol(symbol)
        else:
            symbol_name = stmt.alias if stmt.alias else stmt.module
            symbol = Symbol(
                name=symbol_name,
                kind=SymbolKind.MODULE,
                type_info=f"module {stmt.module}",
                location=location,
            )
            self.symbols.add_symbol(symbol)

    def _collect_use_symbols(self, stmt: UseStatement) -> None:
        """Collect symbols from a use statement."""
        location = self._make_location(stmt.location) if stmt.location else None
        module_path = ".".join(stmt.module_path)

        symbol_name = stmt.alias if stmt.alias else stmt.module_path[-1]
        symbol = Symbol(
            name=symbol_name,
            kind=SymbolKind.MODULE,
            type_info=f"use {module_path}",
            location=location,
        )
        self.symbols.add_symbol(symbol)

    def _collect_for_symbols(self, stmt: ForStatement) -> None:
        """Collect symbols from a for statement."""
        # The loop variable is only in scope within the body
        location = self._make_location(stmt.location) if stmt.location else None

        self.symbols.enter_scope(f"for_{stmt.variable}")

        symbol = Symbol(
            name=stmt.variable,
            kind=SymbolKind.VARIABLE,
            location=location,
        )
        self.symbols.add_symbol(symbol)

        if stmt.body:
            self._collect_block_symbols(stmt.body)

        self.symbols.exit_scope()

    def _collect_if_symbols(self, stmt: IfStatement) -> None:
        """Collect symbols from an if statement."""
        if stmt.then_block:
            self._collect_block_symbols(stmt.then_block)

        for _, elif_block in stmt.elif_clauses:
            if elif_block:
                self._collect_block_symbols(elif_block)

        if stmt.else_block:
            self._collect_block_symbols(stmt.else_block)

    def _collect_while_symbols(self, stmt: WhileStatement) -> None:
        """Collect symbols from a while statement."""
        if stmt.body:
            self._collect_block_symbols(stmt.body)

    def _collect_block_symbols(self, block: Block) -> None:
        """Collect symbols from a block."""
        for stmt in block.statements:
            self._collect_statement_symbols(stmt)

    def _make_location(self, loc) -> Location | None:
        """Convert a source location to an LSP location."""
        if loc is None:
            return None
        return Location(
            uri=self.uri,
            line=loc.line - 1,  # Convert to 0-indexed
            character=loc.column - 1,
        )

    def _format_type(self, type_ann: TypeAnnotation | None) -> str:
        """Format a type annotation as a string."""
        if type_ann is None:
            return "?"

        if isinstance(type_ann, SimpleType):
            return type_ann.name
        elif isinstance(type_ann, GenericType):
            args = ", ".join(self._format_type(arg) for arg in type_ann.type_args)
            return f"{type_ann.base}[{args}]"
        else:
            return str(type_ann)

    # =========================================================================
    # Position-based queries
    # =========================================================================

    def get_completions(self, line: int, character: int) -> list[types.CompletionItem]:
        """
        Get completion items at a position.

        Args:
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            List of completion items
        """
        # Build completion context
        context = self._build_completion_context(line, character)

        # Get visible symbols
        visible_symbols = self.symbols.get_completions_in_scope()

        # Get completions from provider
        return self._completion_provider.get_all_completions(context, visible_symbols)

    def _build_completion_context(self, line: int, character: int) -> CompletionContext:
        """Build a completion context for the given position."""
        prefix = ""
        is_after_dot = False
        is_in_type_position = False
        is_in_import = False
        is_at_line_start = False

        if 0 <= line < len(self.lines):
            line_text = self.lines[line]
            prefix = line_text[:character] if character <= len(line_text) else line_text

            # Check if after a dot
            stripped = prefix.rstrip()
            is_after_dot = stripped.endswith(".")

            # Check if at line start (only whitespace before cursor)
            is_at_line_start = not prefix.strip()

            # Check if in type position (after : or ->)
            is_in_type_position = ": " in prefix or "->" in prefix

            # Check if in import context
            is_in_import = "import " in prefix or "use " in prefix or "from " in prefix

        return CompletionContext(
            line=line,
            character=character,
            trigger_character=None,
            prefix=prefix,
            is_after_dot=is_after_dot,
            is_in_type_position=is_in_type_position,
            is_in_import=is_in_import,
            is_at_line_start=is_at_line_start,
        )

    def get_hover(self, line: int, character: int) -> types.Hover | None:
        """
        Get hover information at a position.

        Args:
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            Hover information or None
        """
        # Find the word at the position
        word, word_range = self._get_word_at_position(line, character)
        if not word:
            return None

        # Look up the symbol
        symbol = self.symbols.lookup(word)
        if symbol:
            return self._create_hover_for_symbol(symbol, word_range)

        # Check if it's a keyword
        from mathviz.compiler.tokens import KEYWORDS
        if word in KEYWORDS:
            return types.Hover(
                contents=types.MarkupContent(
                    kind=types.MarkupKind.Markdown,
                    value=f"**{word}**\n\nMathViz keyword",
                ),
                range=word_range,
            )

        # Check if it's a builtin type
        from mathviz.lsp.completions import BUILTIN_TYPES
        if word in BUILTIN_TYPES:
            return types.Hover(
                contents=types.MarkupContent(
                    kind=types.MarkupKind.Markdown,
                    value=f"**{word}**\n\n{BUILTIN_TYPES[word]}",
                ),
                range=word_range,
            )

        return None

    def _create_hover_for_symbol(
        self, symbol: Symbol, range_: types.Range
    ) -> types.Hover:
        """Create hover content for a symbol."""
        parts: list[str] = []

        # Add signature or type info
        if symbol.signature:
            parts.append(f"```mathviz\n{symbol.signature}\n```")
        elif symbol.type_info:
            kind_name = symbol.kind.name.lower()
            parts.append(f"```mathviz\n({kind_name}) {symbol.name}: {symbol.type_info}\n```")
        else:
            kind_name = symbol.kind.name.lower()
            parts.append(f"```mathviz\n({kind_name}) {symbol.name}\n```")

        # Add documentation
        if symbol.doc:
            parts.append(symbol.doc)

        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value="\n\n".join(parts),
            ),
            range=range_,
        )

    def get_definition(self, line: int, character: int) -> types.Location | None:
        """
        Get the definition location for a symbol at a position.

        Args:
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            Definition location or None
        """
        word, _ = self._get_word_at_position(line, character)
        if not word:
            return None

        symbol = self.symbols.lookup(word)
        if symbol and symbol.location:
            return symbol.location.to_lsp_location()

        return None

    def get_references(
        self, line: int, character: int, include_declaration: bool = True
    ) -> list[types.Location]:
        """
        Get all references to a symbol at a position.

        Args:
            line: 0-indexed line number
            character: 0-indexed character position
            include_declaration: Whether to include the declaration

        Returns:
            List of reference locations
        """
        word, _ = self._get_word_at_position(line, character)
        if not word:
            return []

        symbol = self.symbols.lookup(word)
        if not symbol:
            return []

        references: list[types.Location] = []

        # Add declaration if requested
        if include_declaration and symbol.location:
            references.append(symbol.location.to_lsp_location())

        # Add references
        for ref_loc in self.symbols.get_references(symbol):
            references.append(ref_loc.to_lsp_location())

        # Also search for occurrences of the word in the source
        # (This is a simple text-based search as fallback)
        for i, line_text in enumerate(self.lines):
            start = 0
            while True:
                idx = line_text.find(word, start)
                if idx == -1:
                    break

                # Check that it's a word boundary
                if idx > 0 and (line_text[idx - 1].isalnum() or line_text[idx - 1] == "_"):
                    start = idx + 1
                    continue
                end_idx = idx + len(word)
                if end_idx < len(line_text) and (
                    line_text[end_idx].isalnum() or line_text[end_idx] == "_"
                ):
                    start = idx + 1
                    continue

                loc = types.Location(
                    uri=self.uri,
                    range=types.Range(
                        start=types.Position(line=i, character=idx),
                        end=types.Position(line=i, character=end_idx),
                    ),
                )
                # Avoid duplicates
                if not any(
                    r.uri == loc.uri
                    and r.range.start.line == loc.range.start.line
                    and r.range.start.character == loc.range.start.character
                    for r in references
                ):
                    references.append(loc)

                start = idx + 1

        return references

    def get_document_symbols(self) -> list[types.DocumentSymbol]:
        """
        Get all document symbols for outline view.

        Returns:
            List of document symbols
        """
        # Get top-level symbols
        top_level = self.symbols.scopes.get("", [])
        return [symbol.to_document_symbol() for symbol in top_level]

    def _get_word_at_position(
        self, line: int, character: int
    ) -> tuple[str, types.Range | None]:
        """
        Get the word at a position.

        Args:
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            Tuple of (word, range) or ("", None) if no word found
        """
        if line < 0 or line >= len(self.lines):
            return "", None

        line_text = self.lines[line]
        if character < 0 or character > len(line_text):
            return "", None

        # Find word boundaries
        start = character
        while start > 0 and (line_text[start - 1].isalnum() or line_text[start - 1] == "_"):
            start -= 1

        end = character
        while end < len(line_text) and (line_text[end].isalnum() or line_text[end] == "_"):
            end += 1

        if start == end:
            return "", None

        word = line_text[start:end]
        range_ = types.Range(
            start=types.Position(line=line, character=start),
            end=types.Position(line=line, character=end),
        )

        return word, range_
