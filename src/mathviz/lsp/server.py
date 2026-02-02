"""
MathViz Language Server Protocol (LSP) Server.

This module implements a fully-featured LSP server for the MathViz language
using pygls (Python Language Server). It provides IDE features including:

- Document synchronization (open, change, save, close)
- Diagnostics (errors, warnings)
- Completion suggestions
- Hover information
- Go-to-definition
- Find references
- Rename refactoring
- Document formatting
- Document symbols (outline)

Usage:
    # Start the server in stdio mode (for IDE integration)
    mathviz-lsp

    # Start in TCP mode (for debugging)
    mathviz-lsp --tcp --port 2087
"""

import logging

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from mathviz.lsp.analyzer import DocumentAnalyzer
from mathviz.lsp.formatting import LSPFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mathviz-lsp")


class MathVizLanguageServer(LanguageServer):
    """
    Language Server Protocol implementation for MathViz.

    This class handles LSP requests and notifications, managing document
    state and providing language features through the analyzer.
    """

    def __init__(self) -> None:
        """Initialize the MathViz language server."""
        super().__init__(
            name="mathviz-lsp",
            version="v0.1.0",
        )

        # Document analyzers cache (uri -> analyzer)
        self._analyzers: dict[str, DocumentAnalyzer] = {}

        # Formatter instance
        self._formatter = LSPFormatter()

        # Register all handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all LSP request and notification handlers."""
        # Document synchronization
        self.feature(types.TEXT_DOCUMENT_DID_OPEN)(self._on_did_open)
        self.feature(types.TEXT_DOCUMENT_DID_CHANGE)(self._on_did_change)
        self.feature(types.TEXT_DOCUMENT_DID_SAVE)(self._on_did_save)
        self.feature(types.TEXT_DOCUMENT_DID_CLOSE)(self._on_did_close)

        # Completion
        self.feature(
            types.TEXT_DOCUMENT_COMPLETION,
            types.CompletionOptions(
                trigger_characters=[".", ":", "@", "(", "[", "{", ",", " "],
                resolve_provider=False,
            ),
        )(self._on_completion)

        # Hover
        self.feature(types.TEXT_DOCUMENT_HOVER)(self._on_hover)

        # Go to definition
        self.feature(types.TEXT_DOCUMENT_DEFINITION)(self._on_definition)

        # Find references
        self.feature(types.TEXT_DOCUMENT_REFERENCES)(self._on_references)

        # Document symbols (outline)
        self.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)(self._on_document_symbol)

        # Formatting
        self.feature(types.TEXT_DOCUMENT_FORMATTING)(self._on_formatting)
        self.feature(types.TEXT_DOCUMENT_RANGE_FORMATTING)(self._on_range_formatting)
        self.feature(
            types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
            types.DocumentOnTypeFormattingOptions(
                first_trigger_character="}",
                more_trigger_character=["\n", ";"],
            ),
        )(self._on_type_formatting)

        # Rename
        self.feature(
            types.TEXT_DOCUMENT_RENAME,
            types.RenameOptions(prepare_provider=True),
        )(self._on_rename)
        self.feature(types.TEXT_DOCUMENT_PREPARE_RENAME)(self._on_prepare_rename)

    def _get_analyzer(self, uri: str) -> DocumentAnalyzer | None:
        """Get or create an analyzer for a document."""
        return self._analyzers.get(uri)

    def _analyze_document(self, uri: str, text: str) -> DocumentAnalyzer:
        """Analyze a document and cache the result."""
        analyzer = DocumentAnalyzer(text, uri)
        analyzer.analyze()
        self._analyzers[uri] = analyzer
        return analyzer

    def _publish_diagnostics(self, uri: str, diagnostics: list[types.Diagnostic]) -> None:
        """Publish diagnostics to the client."""
        self.publish_diagnostics(uri, diagnostics)

    # =========================================================================
    # Document Synchronization
    # =========================================================================

    def _on_did_open(self, params: types.DidOpenTextDocumentParams) -> None:
        """Handle document open notification."""
        document = params.text_document
        logger.info(f"Document opened: {document.uri}")

        analyzer = self._analyze_document(document.uri, document.text)
        self._publish_diagnostics(document.uri, analyzer.diagnostics)

    def _on_did_change(self, params: types.DidChangeTextDocumentParams) -> None:
        """Handle document change notification."""
        uri = params.text_document.uri

        # Get the current document text
        doc = self.workspace.get_text_document(uri)
        if doc is None:
            return

        logger.debug(f"Document changed: {uri}")

        analyzer = self._analyze_document(uri, doc.source)
        self._publish_diagnostics(uri, analyzer.diagnostics)

    def _on_did_save(self, params: types.DidSaveTextDocumentParams) -> None:
        """Handle document save notification."""
        uri = params.text_document.uri
        logger.info(f"Document saved: {uri}")

        # Re-analyze on save for completeness
        doc = self.workspace.get_text_document(uri)
        if doc:
            analyzer = self._analyze_document(uri, doc.source)
            self._publish_diagnostics(uri, analyzer.diagnostics)

    def _on_did_close(self, params: types.DidCloseTextDocumentParams) -> None:
        """Handle document close notification."""
        uri = params.text_document.uri
        logger.info(f"Document closed: {uri}")

        # Clean up analyzer
        if uri in self._analyzers:
            del self._analyzers[uri]

        # Clear diagnostics
        self._publish_diagnostics(uri, [])

    # =========================================================================
    # Completion
    # =========================================================================

    def _on_completion(
        self, params: types.CompletionParams
    ) -> types.CompletionList | None:
        """Handle completion request."""
        uri = params.text_document.uri
        position = params.position

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            # Try to analyze the document
            doc = self.workspace.get_text_document(uri)
            if doc is None:
                return None
            analyzer = self._analyze_document(uri, doc.source)

        items = analyzer.get_completions(position.line, position.character)

        return types.CompletionList(
            is_incomplete=False,
            items=items,
        )

    # =========================================================================
    # Hover
    # =========================================================================

    def _on_hover(self, params: types.HoverParams) -> types.Hover | None:
        """Handle hover request."""
        uri = params.text_document.uri
        position = params.position

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            return None

        return analyzer.get_hover(position.line, position.character)

    # =========================================================================
    # Go to Definition
    # =========================================================================

    def _on_definition(
        self, params: types.DefinitionParams
    ) -> types.Location | None:
        """Handle go-to-definition request."""
        uri = params.text_document.uri
        position = params.position

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            return None

        return analyzer.get_definition(position.line, position.character)

    # =========================================================================
    # Find References
    # =========================================================================

    def _on_references(
        self, params: types.ReferenceParams
    ) -> list[types.Location] | None:
        """Handle find-references request."""
        uri = params.text_document.uri
        position = params.position
        include_declaration = params.context.include_declaration

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            return None

        return analyzer.get_references(
            position.line, position.character, include_declaration
        )

    # =========================================================================
    # Document Symbols
    # =========================================================================

    def _on_document_symbol(
        self, params: types.DocumentSymbolParams
    ) -> list[types.DocumentSymbol] | None:
        """Handle document symbols request (for outline view)."""
        uri = params.text_document.uri

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            return None

        return analyzer.get_document_symbols()

    # =========================================================================
    # Formatting
    # =========================================================================

    def _on_formatting(
        self, params: types.DocumentFormattingParams
    ) -> list[types.TextEdit] | None:
        """Handle document formatting request."""
        uri = params.text_document.uri

        doc = self.workspace.get_text_document(uri)
        if doc is None:
            return None

        return self._formatter.format_document(doc.source)

    def _on_range_formatting(
        self, params: types.DocumentRangeFormattingParams
    ) -> list[types.TextEdit] | None:
        """Handle range formatting request."""
        uri = params.text_document.uri
        range_ = params.range

        doc = self.workspace.get_text_document(uri)
        if doc is None:
            return None

        return self._formatter.format_range(
            doc.source,
            range_.start.line,
            range_.start.character,
            range_.end.line,
            range_.end.character,
        )

    def _on_type_formatting(
        self, params: types.DocumentOnTypeFormattingParams
    ) -> list[types.TextEdit] | None:
        """Handle on-type formatting request."""
        uri = params.text_document.uri
        position = params.position
        char = params.ch

        doc = self.workspace.get_text_document(uri)
        if doc is None:
            return None

        return self._formatter.format_on_type(
            doc.source,
            position.line,
            position.character,
            char,
        )

    # =========================================================================
    # Rename
    # =========================================================================

    def _on_prepare_rename(
        self, params: types.PrepareRenameParams
    ) -> types.Range | None:
        """Handle prepare-rename request (validates the rename is possible)."""
        uri = params.text_document.uri
        position = params.position

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            return None

        # Get the word at the position
        word, word_range = analyzer._get_word_at_position(
            position.line, position.character
        )
        if not word or word_range is None:
            return None

        # Check if it's a symbol we can rename
        symbol = analyzer.symbols.lookup(word)
        if symbol is None:
            return None

        return word_range

    def _on_rename(
        self, params: types.RenameParams
    ) -> types.WorkspaceEdit | None:
        """Handle rename request."""
        uri = params.text_document.uri
        position = params.position
        new_name = params.new_name

        analyzer = self._get_analyzer(uri)
        if analyzer is None:
            return None

        # Get all references to the symbol
        references = analyzer.get_references(
            position.line, position.character, include_declaration=True
        )

        if not references:
            return None

        # Build text edits for each reference
        changes: dict[str, list[types.TextEdit]] = {}

        for ref in references:
            ref_uri = ref.uri
            if ref_uri not in changes:
                changes[ref_uri] = []

            changes[ref_uri].append(
                types.TextEdit(
                    range=ref.range,
                    new_text=new_name,
                )
            )

        return types.WorkspaceEdit(changes=changes)


# =============================================================================
# Server Creation and Main Entry Point
# =============================================================================


def create_server() -> MathVizLanguageServer:
    """Create and configure a MathViz language server instance."""
    server = MathVizLanguageServer()

    @server.feature(types.INITIALIZE)
    def on_initialize(
        params: types.InitializeParams,  # noqa: ARG001
    ) -> types.InitializeResult:
        """Handle initialize request."""
        logger.info("Initializing MathViz Language Server")

        return types.InitializeResult(
            capabilities=types.ServerCapabilities(
                # Document synchronization
                text_document_sync=types.TextDocumentSyncOptions(
                    open_close=True,
                    change=types.TextDocumentSyncKind.Full,
                    save=types.SaveOptions(include_text=True),
                ),
                # Completion
                completion_provider=types.CompletionOptions(
                    trigger_characters=[".", ":", "@", "(", "[", "{", ",", " "],
                    resolve_provider=False,
                ),
                # Hover
                hover_provider=True,
                # Go to definition
                definition_provider=True,
                # Find references
                references_provider=True,
                # Document symbols
                document_symbol_provider=True,
                # Formatting
                document_formatting_provider=True,
                document_range_formatting_provider=True,
                document_on_type_formatting_provider=types.DocumentOnTypeFormattingOptions(
                    first_trigger_character="}",
                    more_trigger_character=["\n", ";"],
                ),
                # Rename
                rename_provider=types.RenameOptions(prepare_provider=True),
            ),
            server_info=types.ServerInfo(
                name="mathviz-lsp",
                version="0.1.0",
            ),
        )

    @server.feature(types.INITIALIZED)
    def on_initialized(
        params: types.InitializedParams,  # noqa: ARG001
    ) -> None:
        """Handle initialized notification."""
        logger.info("MathViz Language Server initialized successfully")

    @server.feature(types.SHUTDOWN)
    def on_shutdown(
        params: None,  # noqa: ARG001
    ) -> None:
        """Handle shutdown request."""
        logger.info("Shutting down MathViz Language Server")

    return server


def main() -> None:
    """
    Main entry point for the MathViz language server.

    Starts the server in stdio mode for IDE integration.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="MathViz Language Server",
        prog="mathviz-lsp",
    )
    parser.add_argument(
        "--tcp",
        action="store_true",
        help="Start server in TCP mode instead of stdio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to in TCP mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2087,
        help="Port to listen on in TCP mode (default: 2087)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Configure logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger("mathviz-lsp").setLevel(log_level)

    server = create_server()

    if args.tcp:
        logger.info(f"Starting MathViz LSP in TCP mode on {args.host}:{args.port}")
        server.start_tcp(args.host, args.port)
    else:
        logger.info("Starting MathViz LSP in stdio mode")
        server.start_io()


if __name__ == "__main__":
    main()
