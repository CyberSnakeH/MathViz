"""
Entry point for running the MathViz LSP server as a module.

Usage:
    python -m mathviz.lsp
    python -m mathviz.lsp --tcp --port 2087
"""

from mathviz.lsp.server import main

if __name__ == "__main__":
    main()
