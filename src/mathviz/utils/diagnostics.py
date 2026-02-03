"""
Rust-like Rich Error Diagnostics for MathViz.

This module provides a comprehensive diagnostic system inspired by Rust's
compiler error messages. It transforms basic errors into helpful, educational
diagnostics with source code context, suggestions, and detailed explanations.

Example output:
    error[E0102]: undefined variable 'circel'
      --> example.mviz:5:12
       |
     5 |     let area = circel.radius * PI
       |                ^^^^^^
       |
       = help: did you mean 'circle'?
       = note: 'circle' was defined at line 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# Error Codes Catalog
# =============================================================================


class ErrorCode:
    """
    Centralized catalog of error codes for MathViz diagnostics.

    Error codes are organized by category:
    - E01xx: Type errors
    - E02xx: Syntax errors
    - E03xx: Semantic errors
    - W01xx: Warnings
    """

    # Type errors: E01xx
    E0101 = "E0101"  # type mismatch
    E0102 = "E0102"  # undefined variable
    E0103 = "E0103"  # undefined function
    E0104 = "E0104"  # wrong number of arguments
    E0105 = "E0105"  # incompatible types in operation
    E0106 = "E0106"  # cannot assign to type
    E0107 = "E0107"  # not callable
    E0108 = "E0108"  # not indexable
    E0109 = "E0109"  # not iterable
    E0110 = "E0110"  # member not found
    E0111 = "E0111"  # return type mismatch
    E0112 = "E0112"  # variable redefinition

    # Syntax errors: E02xx
    E0201 = "E0201"  # unexpected token
    E0202 = "E0202"  # unclosed delimiter
    E0203 = "E0203"  # missing token
    E0204 = "E0204"  # invalid expression
    E0205 = "E0205"  # invalid statement
    E0206 = "E0206"  # unterminated string
    E0207 = "E0207"  # invalid number
    E0208 = "E0208"  # unexpected character

    # Semantic errors: E03xx
    E0301 = "E0301"  # break outside loop
    E0302 = "E0302"  # return outside function
    E0303 = "E0303"  # continue outside loop
    E0304 = "E0304"  # invalid import / module not found
    E0305 = "E0305"  # circular dependency
    E0306 = "E0306"  # private member access
    E0307 = "E0307"  # duplicate module
    E0308 = "E0308"  # invalid module path

    # Warnings: W01xx
    W0101 = "W0101"  # unused variable
    W0102 = "W0102"  # unused import
    W0103 = "W0103"  # shadowed variable
    W0104 = "W0104"  # implicit type conversion


# Error code descriptions for documentation
ERROR_DESCRIPTIONS: dict[str, str] = {
    ErrorCode.E0101: "type mismatch",
    ErrorCode.E0102: "undefined variable",
    ErrorCode.E0103: "undefined function",
    ErrorCode.E0104: "wrong number of arguments",
    ErrorCode.E0105: "incompatible types in operation",
    ErrorCode.E0106: "cannot assign to type",
    ErrorCode.E0107: "not callable",
    ErrorCode.E0108: "not indexable",
    ErrorCode.E0109: "not iterable",
    ErrorCode.E0110: "member not found",
    ErrorCode.E0111: "return type mismatch",
    ErrorCode.E0112: "variable redefinition",
    ErrorCode.E0201: "unexpected token",
    ErrorCode.E0202: "unclosed delimiter",
    ErrorCode.E0203: "missing token",
    ErrorCode.E0204: "invalid expression",
    ErrorCode.E0205: "invalid statement",
    ErrorCode.E0206: "unterminated string",
    ErrorCode.E0207: "invalid number",
    ErrorCode.E0208: "unexpected character",
    ErrorCode.E0301: "break outside loop",
    ErrorCode.E0302: "return outside function",
    ErrorCode.E0303: "continue outside loop",
    ErrorCode.E0304: "invalid import / module not found",
    ErrorCode.E0305: "circular dependency",
    ErrorCode.E0306: "private member access",
    ErrorCode.E0307: "duplicate module",
    ErrorCode.E0308: "invalid module path",
    ErrorCode.W0101: "unused variable",
    ErrorCode.W0102: "unused import",
    ErrorCode.W0103: "shadowed variable",
    ErrorCode.W0104: "implicit type conversion",
}


# =============================================================================
# Diagnostic Types
# =============================================================================


class DiagnosticLevel(Enum):
    """Severity level of a diagnostic message."""

    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    HELP = "help"

    def color_code(self) -> str:
        """Get ANSI color code for this level."""
        colors = {
            DiagnosticLevel.ERROR: "\033[91m",  # Red
            DiagnosticLevel.WARNING: "\033[93m",  # Yellow
            DiagnosticLevel.NOTE: "\033[96m",  # Cyan
            DiagnosticLevel.HELP: "\033[92m",  # Green
        }
        return colors.get(self, "")


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """
    A span of source code, representing a range of characters.

    Attributes:
        start_line: 1-indexed starting line number
        start_col: 1-indexed starting column number
        end_line: 1-indexed ending line number
        end_col: 1-indexed ending column number (exclusive)
        filename: Optional filename for display
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    filename: str = "<input>"

    @classmethod
    def from_location(
        cls, line: int, col: int, length: int = 1, filename: str = "<input>"
    ) -> "SourceSpan":
        """Create a span from a single location with a given length."""
        return cls(
            start_line=line,
            start_col=col,
            end_line=line,
            end_col=col + length,
            filename=filename,
        )

    @classmethod
    def single_line(
        cls, line: int, start_col: int, end_col: int, filename: str = "<input>"
    ) -> "SourceSpan":
        """Create a span on a single line."""
        return cls(
            start_line=line,
            start_col=start_col,
            end_line=line,
            end_col=end_col,
            filename=filename,
        )

    def __str__(self) -> str:
        return f"{self.filename}:{self.start_line}:{self.start_col}"

    @property
    def is_multiline(self) -> bool:
        """Check if this span covers multiple lines."""
        return self.start_line != self.end_line

    @property
    def length(self) -> int:
        """Get the length of the span on a single line."""
        if self.is_multiline:
            return 1  # Simplified for multiline
        return max(1, self.end_col - self.start_col)


@dataclass(slots=True)
class DiagnosticLabel:
    """
    A label pointing to a specific span of source code.

    Labels are displayed as underlines or annotations beneath the source code
    to highlight relevant portions of the code.

    Attributes:
        span: The source span this label points to
        message: Optional message to display with the label
        is_primary: Whether this is the primary label (shown with ^^^)
    """

    span: SourceSpan
    message: str = ""
    is_primary: bool = True


@dataclass(slots=True)
class Suggestion:
    """
    A code suggestion that can be applied to fix an error.

    Attributes:
        span: The span of code to replace
        replacement: The suggested replacement text
        message: Description of the suggestion
    """

    span: SourceSpan
    replacement: str
    message: str


@dataclass
class Diagnostic:
    """
    A rich diagnostic message with source context and suggestions.

    This is the main class for representing compiler errors, warnings,
    and other diagnostic messages in a Rust-like format.

    Attributes:
        code: Error code (e.g., "E0102")
        level: Severity level (ERROR, WARNING, NOTE, HELP)
        message: The main diagnostic message
        labels: List of source code labels
        notes: Additional notes to display
        helps: Help messages with suggestions
        suggestions: Code suggestions for fixes
    """

    code: str
    level: DiagnosticLevel
    message: str
    labels: list[DiagnosticLabel] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    helps: list[str] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)

    def render(self, source_code: str, use_color: bool = True) -> str:
        """
        Render this diagnostic as a formatted string.

        Args:
            source_code: The original source code for context
            use_color: Whether to use ANSI color codes

        Returns:
            A formatted multi-line string representation
        """
        lines: list[str] = []
        source_lines = source_code.splitlines()

        # Color helpers
        reset = "\033[0m" if use_color else ""
        bold = "\033[1m" if use_color else ""
        dim = "\033[2m" if use_color else ""
        level_color = self.level.color_code() if use_color else ""
        blue = "\033[94m" if use_color else ""

        # Header line: error[E0102]: undefined variable 'circel'
        level_str = self.level.value
        code_desc = ERROR_DESCRIPTIONS.get(self.code, "")
        if code_desc:
            header = (
                f"{level_color}{bold}{level_str}[{self.code}]{reset}: {bold}{self.message}{reset}"
            )
        else:
            header = f"{level_color}{bold}{level_str}{reset}: {bold}{self.message}{reset}"
        lines.append(header)

        # Location line: --> example.mviz:5:12
        if self.labels:
            primary_label = next((l for l in self.labels if l.is_primary), self.labels[0])
            location_str = f"  {blue}-->{reset} {primary_label.span}"
            lines.append(location_str)

        # Source code context with labels
        if self.labels and source_lines:
            lines.append(f"   {blue}|{reset}")

            # Group labels by line
            labels_by_line: dict[int, list[DiagnosticLabel]] = {}
            for label in self.labels:
                line_num = label.span.start_line
                if line_num not in labels_by_line:
                    labels_by_line[line_num] = []
                labels_by_line[line_num].append(label)

            # Render each line with its labels
            for line_num in sorted(labels_by_line.keys()):
                if 1 <= line_num <= len(source_lines):
                    source_line = source_lines[line_num - 1]
                    line_num_str = f"{line_num:3}"
                    lines.append(f"{blue}{line_num_str} |{reset} {source_line}")

                    # Render underlines and messages
                    for label in labels_by_line[line_num]:
                        underline_char = "^" if label.is_primary else "-"
                        underline_color = level_color if label.is_primary else blue

                        padding = " " * (label.span.start_col - 1)
                        underline = underline_char * label.span.length

                        underline_line = (
                            f"   {blue}|{reset} {padding}{underline_color}{underline}{reset}"
                        )
                        if label.message:
                            underline_line += f" {underline_color}{label.message}{reset}"
                        lines.append(underline_line)

            lines.append(f"   {blue}|{reset}")

        # Notes
        for note in self.notes:
            lines.append(f"   {blue}={reset} {bold}note:{reset} {note}")

        # Help messages
        for help_msg in self.helps:
            green = "\033[92m" if use_color else ""
            lines.append(f"   {blue}={reset} {green}help:{reset} {help_msg}")

        # Suggestions with code
        for suggestion in self.suggestions:
            green = "\033[92m" if use_color else ""
            lines.append(f"   {blue}={reset} {green}suggestion:{reset} {suggestion.message}")
            if suggestion.replacement:
                lines.append(f"   {blue}|{reset}   {suggestion.replacement}")

        return "\n".join(lines)

    def to_simple_message(self) -> str:
        """Get a simple one-line error message for compatibility."""
        return f"[{self.code}] {self.message}"


# =============================================================================
# Diagnostic Builder (Fluent API)
# =============================================================================


class DiagnosticBuilder:
    """
    Fluent builder for constructing Diagnostic objects.

    This provides a convenient API for building diagnostics incrementally:

        emitter.error("E0102", "undefined variable 'x'", span)
            .label(span, "not found in this scope")
            .help("did you mean 'y'?")
            .note("'y' was defined at line 2")
            .emit()
    """

    def __init__(
        self,
        emitter: "DiagnosticEmitter",
        code: str,
        level: DiagnosticLevel,
        message: str,
        primary_span: Optional[SourceSpan] = None,
    ) -> None:
        self._emitter = emitter
        self._code = code
        self._level = level
        self._message = message
        self._labels: list[DiagnosticLabel] = []
        self._notes: list[str] = []
        self._helps: list[str] = []
        self._suggestions: list[Suggestion] = []

        if primary_span:
            self._labels.append(DiagnosticLabel(primary_span, "", True))

    def label(
        self, span: SourceSpan, message: str = "", is_primary: bool = False
    ) -> "DiagnosticBuilder":
        """Add a source code label."""
        self._labels.append(DiagnosticLabel(span, message, is_primary))
        return self

    def primary_label(self, span: SourceSpan, message: str = "") -> "DiagnosticBuilder":
        """Add a primary label (replaces any existing primary)."""
        # Remove existing primary labels
        self._labels = [l for l in self._labels if not l.is_primary]
        self._labels.insert(0, DiagnosticLabel(span, message, True))
        return self

    def secondary_label(self, span: SourceSpan, message: str = "") -> "DiagnosticBuilder":
        """Add a secondary label."""
        self._labels.append(DiagnosticLabel(span, message, False))
        return self

    def note(self, message: str) -> "DiagnosticBuilder":
        """Add a note."""
        self._notes.append(message)
        return self

    def help(self, message: str) -> "DiagnosticBuilder":
        """Add a help message."""
        self._helps.append(message)
        return self

    def suggestion(self, span: SourceSpan, replacement: str, message: str) -> "DiagnosticBuilder":
        """Add a code suggestion."""
        self._suggestions.append(Suggestion(span, replacement, message))
        return self

    def emit(self) -> Diagnostic:
        """Build and emit the diagnostic to the emitter."""
        diagnostic = Diagnostic(
            code=self._code,
            level=self._level,
            message=self._message,
            labels=self._labels,
            notes=self._notes,
            helps=self._helps,
            suggestions=self._suggestions,
        )
        self._emitter.add_diagnostic(diagnostic)
        return diagnostic

    def build(self) -> Diagnostic:
        """Build the diagnostic without emitting."""
        return Diagnostic(
            code=self._code,
            level=self._level,
            message=self._message,
            labels=self._labels,
            notes=self._notes,
            helps=self._helps,
            suggestions=self._suggestions,
        )


# =============================================================================
# Diagnostic Emitter
# =============================================================================


class DiagnosticEmitter:
    """
    Collects and renders diagnostics for a source file.

    The emitter maintains a list of diagnostics and provides methods
    for creating new diagnostics via the fluent builder API.

    Usage:
        emitter = DiagnosticEmitter(source, "example.mviz")
        emitter.error("E0102", "undefined variable 'x'", span).emit()
        for diagnostic in emitter.diagnostics:
            print(diagnostic.render(source))
    """

    def __init__(self, source: str, filename: str = "<input>") -> None:
        """
        Initialize the diagnostic emitter.

        Args:
            source: The source code being compiled
            filename: The filename for error reporting
        """
        self.source = source
        self.filename = filename
        self.source_lines = source.splitlines()
        self.diagnostics: list[Diagnostic] = []

    def add_diagnostic(self, diagnostic: Diagnostic) -> None:
        """Add a diagnostic to the collection."""
        self.diagnostics.append(diagnostic)

    def error(
        self, code: str, message: str, span: Optional[SourceSpan] = None
    ) -> DiagnosticBuilder:
        """Create an error diagnostic builder."""
        return DiagnosticBuilder(self, code, DiagnosticLevel.ERROR, message, span)

    def warning(
        self, code: str, message: str, span: Optional[SourceSpan] = None
    ) -> DiagnosticBuilder:
        """Create a warning diagnostic builder."""
        return DiagnosticBuilder(self, code, DiagnosticLevel.WARNING, message, span)

    def note(self, message: str, span: Optional[SourceSpan] = None) -> DiagnosticBuilder:
        """Create a note diagnostic builder."""
        return DiagnosticBuilder(self, "", DiagnosticLevel.NOTE, message, span)

    def has_errors(self) -> bool:
        """Check if any error diagnostics have been emitted."""
        return any(d.level == DiagnosticLevel.ERROR for d in self.diagnostics)

    def error_count(self) -> int:
        """Count the number of error diagnostics."""
        return sum(1 for d in self.diagnostics if d.level == DiagnosticLevel.ERROR)

    def warning_count(self) -> int:
        """Count the number of warning diagnostics."""
        return sum(1 for d in self.diagnostics if d.level == DiagnosticLevel.WARNING)

    def get_line(self, line_num: int) -> str:
        """Get a source line by number (1-indexed)."""
        if 1 <= line_num <= len(self.source_lines):
            return self.source_lines[line_num - 1]
        return ""

    def render_all(self, use_color: bool = True) -> str:
        """Render all diagnostics as a single string."""
        rendered = []
        for diagnostic in self.diagnostics:
            rendered.append(diagnostic.render(self.source, use_color))
        return "\n\n".join(rendered)

    def clear(self) -> None:
        """Clear all diagnostics."""
        self.diagnostics.clear()


# =============================================================================
# String Similarity (Levenshtein Distance)
# =============================================================================


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The edit distance as an integer
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Use two rows for space efficiency
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def suggest_similar(
    name: str,
    candidates: list[str],
    max_distance: int = 2,
    max_suggestions: int = 3,
) -> list[str]:
    """
    Find similar names from a list of candidates.

    Uses Levenshtein distance to find names that are close to the given name,
    which is useful for "did you mean?" suggestions.

    Args:
        name: The name to find suggestions for
        candidates: List of valid names to compare against
        max_distance: Maximum edit distance to consider (default 2)
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of similar names, sorted by similarity (closest first)
    """
    if not candidates:
        return []

    # Calculate distances for all candidates
    scored = []
    for candidate in candidates:
        # Skip if candidate is too different in length
        if abs(len(candidate) - len(name)) > max_distance:
            continue

        distance = levenshtein_distance(name.lower(), candidate.lower())
        if distance <= max_distance:
            scored.append((candidate, distance))

    # Sort by distance (closest first), then alphabetically for ties
    scored.sort(key=lambda x: (x[1], x[0]))

    return [candidate for candidate, _ in scored[:max_suggestions]]


def suggest_similar_with_context(
    name: str,
    candidates: dict[str, "VariableInfo"],
    max_distance: int = 2,
) -> list[tuple[str, "VariableInfo"]]:
    """
    Find similar names with their definition context.

    Args:
        name: The name to find suggestions for
        candidates: Dictionary mapping names to VariableInfo objects
        max_distance: Maximum edit distance

    Returns:
        List of (name, info) tuples for similar names
    """
    similar_names = suggest_similar(name, list(candidates.keys()), max_distance)
    return [(n, candidates[n]) for n in similar_names if n in candidates]


# =============================================================================
# Variable Information (for tracking definitions)
# =============================================================================


@dataclass
class VariableInfo:
    """
    Information about a variable definition.

    Used to track where variables are defined for better error messages.
    """

    name: str
    type_name: str
    defined_at: SourceSpan
    is_mutable: bool = True
    is_parameter: bool = False

    def definition_note(self) -> str:
        """Generate a note string about where this was defined."""
        kind = "parameter" if self.is_parameter else "variable"
        return f"'{self.name}' was defined as a {kind} at line {self.defined_at.start_line}"


# =============================================================================
# Common Diagnostic Helpers
# =============================================================================


def create_undefined_variable_diagnostic(
    emitter: DiagnosticEmitter,
    name: str,
    span: SourceSpan,
    candidates: list[str],
    definitions: Optional[dict[str, VariableInfo]] = None,
) -> Diagnostic:
    """
    Create a diagnostic for an undefined variable error.

    Args:
        emitter: The diagnostic emitter
        name: The undefined variable name
        span: The source span where it was used
        candidates: List of valid variable names for suggestions
        definitions: Optional dict of variable definitions for context

    Returns:
        The created diagnostic
    """
    builder = emitter.error(
        ErrorCode.E0102,
        f"undefined variable '{name}'",
        span,
    )

    # Add "did you mean?" suggestions
    similar = suggest_similar(name, candidates)
    if similar:
        if len(similar) == 1:
            builder.help(f"did you mean '{similar[0]}'?")
        else:
            suggestions_str = ", ".join(f"'{s}'" for s in similar)
            builder.help(f"did you mean one of: {suggestions_str}?")

        # Add notes about where similar variables were defined
        if definitions:
            for suggestion in similar:
                if suggestion in definitions:
                    info = definitions[suggestion]
                    builder.note(info.definition_note())

    return builder.emit()


def create_type_mismatch_diagnostic(
    emitter: DiagnosticEmitter,
    expected: str,
    actual: str,
    span: SourceSpan,
    context: str = "",
) -> Diagnostic:
    """
    Create a diagnostic for a type mismatch error.

    Args:
        emitter: The diagnostic emitter
        expected: The expected type name
        actual: The actual type name
        span: The source span
        context: Optional context (e.g., "in assignment to 'x'")

    Returns:
        The created diagnostic
    """
    message = f"expected type '{expected}', found '{actual}'"
    if context:
        message = f"{message} {context}"

    builder = emitter.error(ErrorCode.E0101, message, span)

    # Add helpful suggestions for common conversions
    if expected == "Float" and actual == "Int":
        builder.help("use 'float(value)' to convert Int to Float")
    elif expected == "Int" and actual == "Float":
        builder.help("use 'int(value)' to convert Float to Int (truncates decimal)")
    elif expected == "String" and actual in ("Int", "Float", "Bool"):
        builder.help(f"use 'str(value)' to convert {actual} to String")
    elif expected == "Bool" and actual in ("Int", "Float"):
        builder.help("non-zero numbers are truthy; use comparison for explicit boolean")

    return builder.emit()


def create_undefined_function_diagnostic(
    emitter: DiagnosticEmitter,
    name: str,
    span: SourceSpan,
    candidates: list[str],
) -> Diagnostic:
    """
    Create a diagnostic for an undefined function error.
    """
    builder = emitter.error(
        ErrorCode.E0103,
        f"undefined function '{name}'",
        span,
    )

    similar = suggest_similar(name, candidates)
    if similar:
        if len(similar) == 1:
            builder.help(f"did you mean '{similar[0]}'?")
        else:
            suggestions_str = ", ".join(f"'{s}'" for s in similar)
            builder.help(f"did you mean one of: {suggestions_str}?")

    return builder.emit()


def create_wrong_arguments_diagnostic(
    emitter: DiagnosticEmitter,
    func_name: str,
    expected_min: int,
    expected_max: int,
    actual: int,
    span: SourceSpan,
) -> Diagnostic:
    """
    Create a diagnostic for wrong number of arguments.
    """
    if expected_min == expected_max:
        expected_str = str(expected_min)
    else:
        expected_str = f"{expected_min}-{expected_max}"

    builder = emitter.error(
        ErrorCode.E0104,
        f"function '{func_name}' expects {expected_str} argument(s), got {actual}",
        span,
    )

    if actual < expected_min:
        builder.help(f"add {expected_min - actual} more argument(s)")
    else:
        builder.help(f"remove {actual - expected_max} argument(s)")

    return builder.emit()


def create_break_outside_loop_diagnostic(
    emitter: DiagnosticEmitter,
    span: SourceSpan,
) -> Diagnostic:
    """Create a diagnostic for 'break' outside of a loop."""
    return (
        emitter.error(
            ErrorCode.E0301,
            "'break' outside of loop",
            span,
        )
        .help("'break' can only be used inside 'for' or 'while' loops")
        .emit()
    )


def create_return_outside_function_diagnostic(
    emitter: DiagnosticEmitter,
    span: SourceSpan,
) -> Diagnostic:
    """Create a diagnostic for 'return' outside of a function."""
    return (
        emitter.error(
            ErrorCode.E0302,
            "'return' outside of function",
            span,
        )
        .help("'return' can only be used inside function definitions")
        .emit()
    )


def create_unexpected_token_diagnostic(
    emitter: DiagnosticEmitter,
    expected: str,
    found: str,
    span: SourceSpan,
) -> Diagnostic:
    """Create a diagnostic for unexpected token."""
    builder = emitter.error(
        ErrorCode.E0201,
        f"expected {expected}, found '{found}'",
        span,
    )
    return builder.emit()


def create_unclosed_delimiter_diagnostic(
    emitter: DiagnosticEmitter,
    delimiter: str,
    open_span: SourceSpan,
    error_span: SourceSpan,
) -> Diagnostic:
    """Create a diagnostic for unclosed delimiter."""
    builder = emitter.error(
        ErrorCode.E0202,
        f"unclosed delimiter '{delimiter}'",
        error_span,
    )
    builder.secondary_label(open_span, f"unclosed '{delimiter}' starts here")
    builder.help(f"add matching closing '{_matching_delimiter(delimiter)}'")
    return builder.emit()


def _matching_delimiter(opening: str) -> str:
    """Get the matching closing delimiter."""
    matches = {"(": ")", "[": "]", "{": "}", "<": ">"}
    return matches.get(opening, opening)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Error codes
    "ErrorCode",
    "ERROR_DESCRIPTIONS",
    # Core types
    "DiagnosticLevel",
    "SourceSpan",
    "DiagnosticLabel",
    "Suggestion",
    "Diagnostic",
    # Builder and emitter
    "DiagnosticBuilder",
    "DiagnosticEmitter",
    # String similarity
    "levenshtein_distance",
    "suggest_similar",
    "suggest_similar_with_context",
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
