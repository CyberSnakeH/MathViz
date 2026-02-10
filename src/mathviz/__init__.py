"""
MathViz - A domain-specific programming language for mathematical animations.

MathViz is a superset of Python with extended mathematical syntax, providing
native support for mathematical symbols and seamless integration with Manim
for creating animated mathematical visualizations.
"""

from mathviz.compiler import compile_file, compile_source
from mathviz.compiler.codegen import CodeGenerator
from mathviz.compiler.lexer import Lexer
from mathviz.compiler.parser import Parser

__version__ = "0.1.7"
__all__ = [
    "compile_source",
    "compile_file",
    "Lexer",
    "Parser",
    "CodeGenerator",
]
