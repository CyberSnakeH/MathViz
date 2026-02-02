"""
MathViz Utilities Package.

Common utilities for error handling, source locations, and diagnostics.
"""

from mathviz.utils.errors import (
    MathVizError,
    LexerError,
    ParserError,
    CodeGenError,
    SourceLocation,
)
from mathviz.utils.diagnostics import (
    # Error codes
    ErrorCode,
    ERROR_DESCRIPTIONS,
    # Core diagnostic types
    DiagnosticLevel,
    SourceSpan,
    DiagnosticLabel,
    Suggestion,
    Diagnostic,
    # Builder and emitter
    DiagnosticBuilder,
    DiagnosticEmitter,
    # String similarity utilities
    levenshtein_distance,
    suggest_similar,
    # Variable info for tracking definitions
    VariableInfo,
    # Helper functions for common diagnostics
    create_undefined_variable_diagnostic,
    create_type_mismatch_diagnostic,
    create_undefined_function_diagnostic,
    create_wrong_arguments_diagnostic,
    create_break_outside_loop_diagnostic,
    create_return_outside_function_diagnostic,
    create_unexpected_token_diagnostic,
    create_unclosed_delimiter_diagnostic,
)

__all__ = [
    # Errors
    "MathVizError",
    "LexerError",
    "ParserError",
    "CodeGenError",
    "SourceLocation",
    # Error codes
    "ErrorCode",
    "ERROR_DESCRIPTIONS",
    # Core diagnostic types
    "DiagnosticLevel",
    "SourceSpan",
    "DiagnosticLabel",
    "Suggestion",
    "Diagnostic",
    # Builder and emitter
    "DiagnosticBuilder",
    "DiagnosticEmitter",
    # String similarity utilities
    "levenshtein_distance",
    "suggest_similar",
    # Variable info
    "VariableInfo",
    # Helper functions
    "create_undefined_variable_diagnostic",
    "create_type_mismatch_diagnostic",
    "create_undefined_function_diagnostic",
    "create_wrong_arguments_diagnostic",
    "create_break_outside_loop_diagnostic",
    "create_return_outside_function_diagnostic",
    "create_unexpected_token_diagnostic",
    "create_unclosed_delimiter_diagnostic",
]
