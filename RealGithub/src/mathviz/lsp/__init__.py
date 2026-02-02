"""
MathViz Language Server Protocol (LSP) implementation.

This package provides a fully-featured LSP server for the MathViz language,
enabling IDE features such as:
- Autocomplete suggestions
- Go-to-definition
- Hover information with type details
- Error diagnostics
- Code formatting
- Find references
- Rename refactoring

The server integrates with the MathViz compiler infrastructure to provide
accurate, context-aware language support.

Usage:
    # Start the LSP server (stdio mode)
    mathviz-lsp

    # Or run as a module
    python -m mathviz.lsp
"""

from mathviz.lsp.server import MathVizLanguageServer, main

__all__ = [
    "MathVizLanguageServer",
    "main",
]
